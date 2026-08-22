"""Two-stage instance retrieval. The product, in one class.

    s = Searcher(index_dir)
    resp = s.query("photo_abc", BoxXYXY(...))        # -> schemas.QueryResponse
    s.feedback(resp.query_id, "photo_xyz", True)     # confirm  -> new exemplar
    s.feedback(resp.query_id, "photo_bad", False)    # reject   -> blacklist

STAGE 1 (recall, cheap)
    The user's box becomes its own image and is embedded with dinov2-base, at
    three dilations so a tight box and a generous one both get a shot at the
    indexed multi-scale regions. hnswlib returns the best-matching region per
    photo. Fast, and deliberately permissive -- these are CANDIDATES.

STAGE 2 (precision, expensive)
    Each candidate goes through SIFT + ratio test + RANSAC (verify.py). Only a
    geometrically consistent correspondence set counts as a match. Inlier count
    is the confidence and is passed through to the UI unmodified.

PRESENTATION
    Verified matches sort oldest -> newest for the timeline. `last_seen` is the
    most recent VERIFIED match -- never merely the highest-ranked or most recent
    candidate, which would be a plausible-sounding lie about where your passport
    is.
"""

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.app.indexing import features as F
from backend.app.indexing.store import IndexStore, open_image
from backend.app.retrieval import verify as V
from backend.app.retrieval.objects import ObjectStore
from backend.app.schemas import BoxXYXY, Match, PhotoRecord, QueryResponse

# ---------------------------------------------------------------------------
# THE OPERATING POINT. Chosen off the precision/recall sweep in eval.py over
# 168 true positives and 672 lookalike/unrelated candidates. Not guessed.
#
#   accept  <=>  inliers >= MIN_INLIERS and inlier_fraction >= MIN_INLIER_FRACTION
#                (or inliers >= STRONG_INLIERS, see below)
#
# Why two gates: the raw inlier count is not scale-invariant. A texture-rich
# object at 30% of the frame gives 93-157 inliers; the same pipeline on a flat
# blurred object at 16% gives 0-12. A single absolute threshold that keeps the
# flat object also admits lookalikes of the textured one. The fraction fixes
# that and lets the absolute floor drop to 6. Grid search over both gates:
# this pair is the highest-recall setting with ZERO false positives on all
# three suites (recall 1.000 / 0.982 / 0.375).
MIN_INLIERS = 6
# 0.40 -> 0.45, re-swept during integration after the query/candidate scale fix
# in verify.crop_features, over 3408 synthetic candidate pairs PLUS 29 real 4K
# video frames. 0.45 is free: identical recall (0.607 overall, 0.214 on the hard
# suite), all 29 real frames still accepted, and it removes the one same-class
# false positive 0.40 let through. Same-class-different-instance acceptance is
# the number that decides whether this product works, so it gets the tie-break.
MIN_INLIER_FRACTION = 0.45

# Escape hatch: in a genuinely cluttered real photo the ratio test throws
# correspondences all over the background, so a true match could sit below the
# fraction gate on sheer match volume. An inlier count far above anything a
# lookalike has produced is accepted on its own.
#
# RAISED 20 -> 30 during integration, on evidence. Swept over all 3408
# candidate pairs of a 72-photo index holding BOTH eval suites (easy + hard,
# cross-folder same-class traps included):
#
#   STRONG   recall   FP same-class   FP unrelated
#      20    0.613          0              4
#      30    0.613          0              2      <- free: same recall, half the FPs
#      40    0.613          0              2
#    off     0.607          0              2
#
# At 20, two DEGRADED unrelated scenes (blur + JPEG q55) reached 20 and 24
# inliers with a high inlier fraction, so a fraction floor on this gate does
# not help -- only the count does. The gate still earns its 0.6 pt of recall
# over turning it off entirely.
STRONG_INLIERS = 30

# Backwards-compatible single knob: callers that just want "how strict" can set
# this, and it moves MIN_INLIERS.
INLIER_THRESHOLD = MIN_INLIERS

# Stage 1 is allowed to be sloppy; stage 2 pays for it. 200 candidates is ~2 s
# of SIFT matching on an M-series laptop with the descriptor cache warm.
DEFAULT_TOP_K = 200

# Query-side dilations of the user's box (fraction of box size added on each
# side). A hand-drawn box is usually a bit tight or a bit loose; embedding all
# three costs one extra forward pass and measurably helps recall.
QUERY_DILATIONS = (0.0, 0.15, 0.4)

WORKERS = min(8, (os.cpu_count() or 4))


@dataclass
class Candidate:
    """A stage-1 hit after stage-2 has ruled on it. Kept separate from `Match`
    so eval.py can see the rejects too -- Match only exists for things that
    passed."""

    photo: PhotoRecord
    recall_score: float
    inliers: int
    raw_matches: int
    box: tuple[float, float, float, float] | None
    scale: float = 0.0
    detections: int = 0
    """Distinct, non-overlapping, accepted detections of the object IN THIS ONE
    PHOTO. 2 is proof the user owns more than one of the item -- see
    duplicate_warning()."""


def dilate(box, frac: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    dx, dy = (x2 - x1) * frac, (y2 - y1) * frac
    return (max(0.0, x1 - dx), max(0.0, y1 - dy), min(1.0, x2 + dx), min(1.0, y2 + dy))


def duplicate_warning(cands: list[Candidate]) -> str | None:
    """Hard evidence, from the pixels, that this object is not unique.

    If any single photo contains two non-overlapping accepted detections, the
    user demonstrably owns at least two of the item, and everything downstream
    -- especially "last seen" -- is about a SET, not a thing. Returns a string
    for the UI, or None. It is deliberately silent when there is no evidence:
    twins that never share a frame are undetectable and we do not guess.
    """
    dupes = [c for c in cands if c.detections >= 2]
    if not dupes:
        return None
    n = max(c.detections for c in dupes)
    return (
        f"{n} of these appear together in {Path(dupes[0].photo.path).name}"
        f"{' and ' + str(len(dupes) - 1) + ' other photo(s)' if len(dupes) > 1 else ''}. "
        "You own more than one; matches below may be different copies, and "
        "'last seen' is the last time ANY copy was photographed."
    )


class Searcher:
    def __init__(
        self,
        index_dir: str | Path | None = None,
        inlier_threshold: int = MIN_INLIERS,
        min_inlier_fraction: float = MIN_INLIER_FRACTION,
        embedder: F.Embedder | None = None,
    ):
        self.store = IndexStore(index_dir)
        self.objects = ObjectStore(self.store.root)
        self.inlier_threshold = inlier_threshold
        self.min_inlier_fraction = min_inlier_fraction
        self._embedder = embedder

    def accept(self, inliers: int, matches: int) -> bool:
        """The operating point, in one place. See the constants above."""
        if inliers >= STRONG_INLIERS:
            return True
        return inliers >= self.inlier_threshold and (
            inliers / max(matches, 1) >= self.min_inlier_fraction
        )

    @property
    def embedder(self) -> F.Embedder:
        if self._embedder is None:
            self._embedder = F.Embedder()
        return self._embedder

    # -- exemplars -------------------------------------------------------

    def _exemplar_boxes(self, source_photo_id, box, object_id) -> list[tuple[str, tuple]]:
        """(photo_id, box) pairs to use as queries: the fresh box plus anything
        the user has confirmed for this object before."""
        out = [(source_photo_id, tuple(float(v) for v in box))]
        if object_id and object_id in self.objects.objects:
            for e in self.objects.get(object_id).exemplars:
                if e["photo_id"] in self.store.photos and (e["photo_id"], tuple(e["box"])) not in out:
                    out.append((e["photo_id"], tuple(e["box"])))
        return out

    # -- stage 1 ---------------------------------------------------------

    def recall(self, exemplars, top_k: int):
        """-> list[RegionHit], best region per photo, descending score."""
        descs = []
        for pid, box in exemplars:
            img = open_image(self.store.photos[pid].path)
            boxes = np.array([dilate(box, d) for d in QUERY_DILATIONS], dtype=np.float32)
            descs.append(self.embedder.embed_crops(img, boxes))
        return self.store.search(np.concatenate(descs, axis=0), top_k_photos=top_k)

    # -- stage 1 + 2 -----------------------------------------------------

    def search(
        self,
        source_photo_id: str,
        box,
        top_k: int = DEFAULT_TOP_K,
        object_id: str | None = None,
        full_affine: bool = False,
    ) -> list[Candidate]:
        exemplars = self._exemplar_boxes(source_photo_id, box, object_id)
        hits = self.recall(exemplars, top_k)

        qfeats = []
        for pid, bx in exemplars:
            xy, desc, size = V.crop_features(open_image(self.store.photos[pid].path), bx)
            if len(desc):
                qfeats.append((xy, desc, size))

        rejected = set(self.objects.get(object_id).rejected) if object_id else set()
        exemplar_ids = {pid for pid, _ in exemplars}

        todo = [h for h in hits if h.photo_id != source_photo_id and h.photo_id not in rejected]

        def verify_one(hit) -> Candidate:
            entry = self.store.photos[hit.photo_id]
            c_xy, c_desc = self.store.sift_for(hit.photo_id)
            best = [V.Verification(0, 0, None)]
            for xy, desc, size in qfeats:
                dets = V.verify_all(xy, desc, size, c_xy, c_desc, (entry.width, entry.height),
                                    full_affine=full_affine)
                if dets[0].inliers > best[0].inliers:
                    best = dets
            n_det = sum(1 for d in best if self.accept(d.inliers, d.matches))
            c = Candidate(entry.record, hit.score, best[0].inliers, best[0].matches,
                          best[0].box, best[0].scale, n_det)
            # An already-confirmed exemplar is a match by definition; don't let
            # a weak re-verification demote what the user personally told us.
            if hit.photo_id in exemplar_ids:
                c.inliers = max(c.inliers, STRONG_INLIERS)
                if c.box is None:
                    c.box = tuple(next(b for p, b in exemplars if p == hit.photo_id))
            return c

        # Stage 2 is the whole query cost and every candidate is independent.
        # OpenCV's matcher and RANSAC release the GIL, so threads are real
        # parallelism here: measured 3.46 s -> 0.65 s median on a 183-photo
        # library of 12 MP photos. (Cost: RANSAC's global RNG is now consumed in
        # a nondeterministic order, so a borderline candidate can flip between
        # runs. That fragility was already there -- see verify.py -- this only
        # makes it visible.)
        if len(todo) <= 4:
            return [verify_one(h) for h in todo]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            return list(pool.map(verify_one, todo))

    def query(
        self,
        source_photo_id: str,
        box: BoxXYXY | tuple,
        top_k: int = DEFAULT_TOP_K,
        object_id: str | None = None,
    ) -> QueryResponse:
        b = (box.x1, box.y1, box.x2, box.y2) if isinstance(box, BoxXYXY) else tuple(box)
        return self._respond(source_photo_id, b, self.search(source_photo_id, b, top_k, object_id),
                             object_id)

    def _respond(self, source_photo_id, b, cands, object_id) -> QueryResponse:
        matches = [
            Match(
                photo_id=c.photo.id,
                score=c.recall_score,
                inlier_count=c.inliers,
                verified=True,
                bbox=BoxXYXY(x1=c.box[0], y1=c.box[1], x2=c.box[2], y2=c.box[3]),
                taken_at=c.photo.taken_at,
                gps=c.photo.gps,
            )
            for c in cands
            if c.box is not None and self.accept(c.inliers, c.raw_matches)
        ]

        # Timeline: oldest -> newest. Undated photos sort to the end rather than
        # pretending to a position they don't have.
        matches.sort(key=lambda m: (m.taken_at is None, m.taken_at or 0))
        dated = [m for m in matches if m.taken_at is not None]
        last_seen = max(dated, key=lambda m: m.taken_at) if dated else (matches[-1] if matches else None)

        qid = object_id or uuid.uuid4().hex
        mem = self.objects.get(qid)
        if not mem.exemplars:
            # Seed the object with the crop the user drew, so the first confirm
            # extends a model instead of creating one from nothing.
            self.objects.add_exemplar(qid, source_photo_id, b)
            self.objects.save()
        return QueryResponse(query_id=qid, matches=matches, last_seen=last_seen)

    # -- feedback --------------------------------------------------------

    def query_detailed(
        self,
        source_photo_id: str,
        box: BoxXYXY | tuple,
        top_k: int = DEFAULT_TOP_K,
        object_id: str | None = None,
    ) -> tuple[QueryResponse, list[Candidate], str | None]:
        """`query()` plus the rejected candidates and a duplicate warning.

        The API/UI layer wants all three: the timeline, the near-misses (useful
        for a "show weak matches too" affordance), and whether this object
        appears to exist in more than one copy.
        """
        b = (box.x1, box.y1, box.x2, box.y2) if isinstance(box, BoxXYXY) else tuple(box)
        cands = self.search(source_photo_id, b, top_k, object_id)
        return self._respond(source_photo_id, b, cands, object_id), cands, duplicate_warning(cands)

    def feedback(self, query_id: str, photo_id: str, confirmed: bool, box=None) -> int:
        """One tap. Returns the object's exemplar count ("learned from N views").

        `box` should be the Match.bbox the user is confirming. Without it we
        re-verify against the seed exemplar to find the region -- storing the
        whole photo as an exemplar would poison the object's descriptor with
        background and is never done.
        """
        if not confirmed:
            self.objects.reject(query_id, photo_id)
            self.objects.save()
            return len(self.objects.get(query_id).exemplars)

        if box is None:
            mem = self.objects.get(query_id)
            if not mem.exemplars:
                raise ValueError("unknown object; pass box")
            seed = mem.exemplars[0]
            xy, desc, size = V.crop_features(open_image(self.store.photos[seed["photo_id"]].path),
                                             seed["box"])
            entry = self.store.photos[photo_id]
            c_xy, c_desc = self.store.sift_for(photo_id)
            v = V.verify(xy, desc, size, c_xy, c_desc, (entry.width, entry.height))
            if v.box is None:
                raise ValueError("cannot locate the object in that photo; pass box explicitly")
            box = v.box
        b = (box.x1, box.y1, box.x2, box.y2) if isinstance(box, BoxXYXY) else tuple(box)
        self.objects.add_exemplar(query_id, photo_id, b)
        self.objects.save()
        return len(self.objects.get(query_id).exemplars)
