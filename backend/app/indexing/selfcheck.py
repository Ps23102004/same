"""Runnable self-check: `uv run python -m backend.app.indexing.selfcheck [folder]`

Indexes a folder twice and asserts the things that would otherwise rot silently:
region geometry, descriptor normalization, persistence, incrementality, and the
one that decides whether the product exists at all — that a query crop of ONE
specific object outranks same-class-different-instance photos.

Stage 1 (this module's index) is RECALL. Stage 2 (RANSAC, in
backend/app/retrieval) is PRECISION. The check therefore holds stage 1 to a
loose bar and holds the cached stage-2 features to a strict separation bar.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from . import Embedder, IndexStore, index_folder, region_boxes, sift_features
from .features import GRID, N_REGIONS
from .store import open_image

ROOT = Path(__file__).resolve().parents[3]


def locate_object(img) -> np.ndarray:
    """Normalized xyxy box of the synthetic sprite: the largest high-local-variance
    blob. Exists only to drive this check on the synthetic sample set — the real
    product uses the box the user drags."""
    g = np.asarray(img.convert("L"), np.float32)
    var = cv2.boxFilter(g * g, -1, (9, 9)) - cv2.boxFilter(g, -1, (9, 9)) ** 2
    mask = (var > np.percentile(var, 88)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    _, _, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    x, y, bw, bh = st[1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])), :4]
    h, w = g.shape
    return np.array([x / w, y / h, (x + bw) / w, (y + bh) / h], np.float32)


def ransac_inliers(qxy, qd, xy, d) -> int:
    if len(qd) < 2 or len(d) < 2:
        return 0
    good = [a for a, b in cv2.BFMatcher(cv2.NORM_L2).knnMatch(qd, d, k=2)
            if a.distance < 0.8 * b.distance]
    if len(good) < 4:
        return 0
    src = np.float32([qxy[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([xy[m.trainIdx] for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def main(folder: Path) -> int:
    boxes = region_boxes()
    assert boxes.shape == (N_REGIONS, 4), boxes.shape
    assert (boxes >= 0).all() and (boxes <= 1).all(), "region boxes must be normalized"
    assert (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all()
    assert tuple(boxes[0]) == (0.0, 0.0, 1.0, 1.0), "region 0 must be the whole image"
    print(f"[ok] region layout: {N_REGIONS} regions over a {GRID}x{GRID} patch grid")

    tmp = Path(tempfile.mkdtemp(prefix="same-selfcheck-"))
    try:
        emb = Embedder()
        print(f"[ok] dinov2 loaded on device={emb.device}")

        s1 = index_folder(folder, index_dir=tmp, embedder=emb)
        print(f"[ok] pass 1: {s1.photos_indexed} indexed in {s1.seconds:.2f}s "
              f"= {s1.images_per_sec:.2f} img/s, {s1.index_bytes / 1e6:.1f} MB on disk")
        assert s1.photos_indexed > 0

        s2 = index_folder(folder, index_dir=tmp, embedder=emb)
        print(f"[ok] pass 2 (incremental): {s2.photos_indexed} indexed, "
              f"{s2.photos_skipped} skipped, {s2.seconds:.2f}s")
        assert s2.photos_indexed == 0, "re-run must not re-embed unchanged files"
        assert s2.photos_skipped == s1.total_photos
        assert s2.index_bytes == s1.index_bytes, "incremental run must not grow the index"

        store = IndexStore(tmp)
        assert len(store.photos) == s1.total_photos
        v = np.asarray(store.regions[0], np.float32)
        assert abs(np.linalg.norm(v) - 1.0) < 2e-3, "descriptors must be unit norm"
        assert store.rows == s1.total_photos * N_REGIONS
        print(f"[ok] reopened from disk: {len(store.photos)} photos, "
              f"{store.rows} region vectors, unit-norm")

        by_name = {Path(e.path).name: e for e in store.photos.values()}
        if "A_00.jpg" not in by_name:
            print("[--] not the synthetic sample set; skipping the retrieval checks")
            return 0

        # evalset.py names instances A/B/C, the pixel-identical twin T, and the
        # object-free scenes `unrelated`. (These were `target_`/`distractor_`
        # once; everything below asked for the old names, so the whole retrieval
        # half of this selfcheck silently skipped instead of running.)
        # T counts as the same instance for STAGE 1 -- it is the same pixels, so
        # ranking it highly is correct -- but it is left out of the stage-2
        # separation test, where it would sit on the wrong side of a boundary
        # that is supposed to be about different objects.
        SAME = ("A_", "T_")
        OTHER = ("B_", "C_", "unrelated_")

        # ---- stage 1: instance recall, not class recall --------------------
        q = by_name["A_00.jpg"]
        img = open_image(q.path)
        w, h = img.size
        qbox = locate_object(img)
        area = (qbox[2] - qbox[0]) * (qbox[3] - qbox[1])
        assert 0.02 < area < 0.5, f"object detector produced a silly box {qbox}"
        print(f"[ok] located the object at {np.round(qbox, 3).tolist()} "
              f"— stand-in for the user's drag")

        qv = emb.embed_crops(img, qbox[None, :])[0]
        hits = [x for x in store.search(qv, top_k_photos=24) if x.photo_id != q.id]
        names = [Path(store.photos[x.photo_id].path).name for x in hits]
        p7 = sum(n.startswith(SAME) for n in names[:7])
        print(f"[--] stage-1 top-7: {names[:7]}")
        assert names[0].startswith(SAME), f"top-1 recall hit is not the instance: {names[0]}"
        assert p7 >= 3, f"stage-1 recall too weak: {p7}/7"
        print(f"[ok] stage-1 recall: top-1 is the same instance, precision@7 = {p7}/7 "
              f"(7 same-instance photos exist; the rest is stage-2's job)")

        # ---- stage 2: the cached features must separate instance from class -
        crop = img.crop((int(qbox[0] * w), int(qbox[1] * h), int(qbox[2] * w), int(qbox[3] * h)))
        qxy, qd = sift_features(crop)
        tgt = [ransac_inliers(qxy, qd, *store.sift_for(by_name[n].id))
               for n in sorted(by_name) if n.startswith("A_") and n != "A_00.jpg"]
        dis = [ransac_inliers(qxy, qd, *store.sift_for(by_name[n].id))
               for n in sorted(by_name) if n.startswith(OTHER)]
        print(f"[--] query crop SIFT keypoints: {len(qxy)}")
        print(f"[--] stage-2 inliers, same instance      ({len(tgt):2d}): {tgt}")
        print(f"[--] stage-2 inliers, same class/other   ({len(dis):2d}): {dis}")
        # Assert a decision boundary EXISTS, and report where it falls, rather
        # than asserting a number picked after seeing the data.
        boundary = max(dis) + 1
        clean = sum(t >= boundary for t in tgt)
        assert clean >= 3, (
            f"no inlier threshold separates instance from class: "
            f"same-instance {tgt} vs same-class {dis}"
        )
        print(f"[ok] stage-2 features separate: at inliers >= {boundary}, "
              f"{clean}/{len(tgt)} same-instance photos pass and "
              f"0/{len(dis)} same-class impostors do")
        assert (store.sift_dir / f"{by_name['A_01.jpg'].id}.npz").exists()
        print("[ok] SIFT cache persisted under .index/sift/")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "sample_photos"))
