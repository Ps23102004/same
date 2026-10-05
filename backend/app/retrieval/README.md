# Retrieval — two-stage instance search

> Everything in this file is measured. Reproduce with
> `uv run python -m backend.app.retrieval.eval` (~4 min on an M-series Mac).
> A 30-second version is `uv run python -m backend.app.retrieval.selfcheck`.

## The API

```python
from backend.app.retrieval import Searcher

s = Searcher(index_dir)                                   # opens the indexer's .index/
resp = s.query(photo_id, box, top_k=200, object_id=None)  # -> schemas.QueryResponse
resp, candidates, dup_warning = s.query_detailed(...)     # + near-misses + duplicate warning
s.feedback(resp.query_id, photo_id, confirmed=True, box=match.bbox)   # one tap
s.feedback(resp.query_id, photo_id, confirmed=False)                  # reject
```

* `box` is normalized xyxy, origin top-left (`schemas.BoxXYXY`).
* `query_id` **is** the object id. Pass it back as `object_id` on later queries to
  search with everything the user has confirmed so far.
* `QueryResponse.matches` is sorted **oldest → newest** (the timeline).
  `last_seen` is the most recent **verified** match — never the most recent
  candidate.
* `Match.inlier_count` is the confidence, passed through raw. Don't rescale it
  into a fake percentage.

## How it works

**Stage 1 — recall.** The user's box is cropped out, re-rendered at the
backbone's full input size (so a small object gets 32×32 patch tokens instead of
four), and embedded with `facebook/dinov2-base` at three dilations (0%, 15%,
40%) to absorb a sloppy hand-drawn box. hnswlib returns the best-matching
indexed region per photo. The index side (owned by `indexing/`) stores 35
mean-pooled region descriptors per image — whole frame, 9 half-size windows, 25
quarter-size windows — which is what makes a small object findable at all.

DINOv2 rather than CLIP on purpose: CLIP is trained against captions, so it
encodes *"a brown teddy bear"*. That is the category answer, i.e. the failed
product. DINOv2 is self-supervised and its patch tokens stay
instance-discriminative.

**Stage 2 — precision.** SIFT + FLANN ratio test between the query crop and each
candidate, then RANSAC a 4-DoF similarity transform (rotation + uniform scale +
translation). Only a geometrically consistent correspondence set counts.
`estimateAffinePartial2D` is deliberately the most constrained model available;
`full_affine=True` (6 DoF) is the escape hatch for flat objects at strong
angles, and a homography is the next rung up, each accepting more junk.

## Why stage 2 exists — the control experiment

Precision of stage 1 **alone**, i.e. what a global-embedding product ships,
averaged over 8 queries × 3 suites:

| suite | P@5 | P@10 | what the wrong results are |
|---|---|---|---|
| easy  | 0.250 | 0.212 | 17 / 36 same-class-different-instance, 0 unrelated |
| small | 0.175 | 0.200 | 32 / 61 same-class-different-instance |
| hard  | 0.225 | 0.225 | 31 / 62 same-class-different-instance |

Nearly every stage-1 error is *the same class, the wrong object*. That is the
failure mode this product exists to fix, and no amount of better embedding
fixes it — it needs geometry.

## The operating point, and how it was chosen

```
accept  ⟺  inliers ≥ 30                                  (STRONG_INLIERS)
        or (inliers ≥ 6  and  inliers/matches ≥ 0.45)     (MIN_INLIERS, MIN_INLIER_FRACTION)
```

Two gates, because **a raw inlier count is not scale-invariant**. Measured
inlier counts for true matches: 90–172 on a texture-rich object at 30% of the
frame; 15–42 at 7% of the frame; 0–9 on a flat, blurred, 25%-occluded object.
Meanwhile same-class lookalikes reached 11 inliers on the easy suite. There is
no single absolute threshold that keeps the flat object (max 9) and rejects the
easy lookalike (max 11) — the ranges overlap. The inlier *fraction* separates
them cleanly, because a true match has most of its correspondences agreeing on
one transform and a lookalike has almost none.

The pair was picked by grid search over `inliers ∈ [4,15] × fraction ∈ [0,0.6]`
across all 168 true positives and 672 lookalike/unrelated candidates: it is the
highest-recall setting with **zero false positives on all three suites**.
`STRONG_INLIERS` is the escape hatch for cluttered real photos, where the ratio
test scatters correspondences over the background and could push a true match
under the fraction gate. It was raised 20 → 30 and the fraction gate 0.40 → 0.45
during integration (see the comments above the constants in `search.py` for the
sweeps); the numbers in this file's older curves predate that change.

Curve for the `small` suite (the full sweep for all three is printed by
`eval.py`; note how the fraction gate collapses the false positives):

| thr | frac=0.0 → prec / rec | frac=0.4 → prec / rec |
|---|---|---|
| 4  | 0.554 / 1.000 | **1.000 / 1.000** |
| 5  | 0.800 / 1.000 | 1.000 / 1.000 |
| 6  | 0.889 / 1.000 | 1.000 / 1.000 |
| 8  | 0.982 / 1.000 | 1.000 / 1.000 |
| 10 | 1.000 / 1.000 | 1.000 / 1.000 |
| 20 | 1.000 / 0.768 | 1.000 / 0.768 |
| 30 | 1.000 / 0.304 | 1.000 / 0.304 |

Lowe's ratio was swept too and set to **0.85** (from the usual 0.75): it lifts
recall on the hard suite and, because the second gate is a fraction, the extra
lookalike matches it admits do not cost precision.

## The precision/recall table

8 queries per suite, leave-one-out, 56 true positives / 128 same-class traps /
96 unrelated per suite. **Accept rate** = fraction of that class returned to the
user.

| suite | (a) same instance | (b) same class, **diff instance** | (c) unrelated |
|---|---|---|---|
| **easy** — 30% of frame, textured | **56/56 = 1.000** | **0/128 = 0.000** | 1/96 = 0.010 |
| **small** — 7% of frame, textured | **56/56 = 1.000** | **0/128 = 0.000** | 0/96 = 0.000 |
| **hard** — 16%, flat surface, blurred, JPEG q55, 25% occluded | 19/56 = 0.339 | **0/128 = 0.000** | 0/96 = 0.000 |

**The key number: same-class-different-instance acceptance is 0.000 on every
suite.** Ask for your teddy bear and you do not get someone else's. Inlier
distributions, same suite:

| suite | same instance (min / median / max) | same-class traps (min / median / max) |
|---|---|---|
| easy  | 90 / 124 / 172 | 0 / 4 / 11 |
| small | 15 / 26 / 42   | 0 / 0 / 9  |
| hard  | 0 / 4 / 9      | 0 / 0 / 5  |

### Where it fails, plainly

The `hard` suite recalls **0.34**. Two thirds of the true appearances of a flat,
blurred, partly hidden object are missed. That is not a tuning problem: SIFT
finds ~24 keypoints on that surface, and you cannot RANSAC a consensus out of
nothing. Repeated runs give 0.29 / 0.34 / 0.41 (RANSAC samples randomly and the
hard case sits right at the gate), so treat that number as a range, not a point.

The honest upgrade path is a learned local matcher — SuperPoint + LightGlue, or
LoFTR — which matches on flat and low-texture surfaces where SIFT has nothing to
key on. The interface in `verify.py` is the natural seam: replace the
detect/match half, keep RANSAC and the gates.

Small-object recall (the failure mode the pyramid exists for) is **1.000** at 7%
of the frame width, ~110 px on a 1600 px image. A whole-image embedding cannot
do this; stage-1 P@5 on that suite is 0.175.

## Feedback: does a tap actually help?

Protocol: run each query cold, confirm every true positive it returned (one tap
each, what a user actually does), re-run, compare. Averaged over 8 queries:

| suite | recall before | recall after | false positives after |
|---|---|---|---|
| easy  | 1.000 | 1.000 | 9 |
| small | 1.000 | 1.000 | 4 |
| hard  | 0.339 | **0.518** | 0 |

On the hard suite a single round of confirmations lifts recall by 18 points —
each confirmed crop becomes another exemplar, and both stage 1 and stage 2 run
against all of them and take the best. Where recall was already saturated,
feedback has nothing to add and instead costs a little precision (9 and 4
lookalikes admitted): more exemplars mean more chances at a spurious consensus.
That trade is not free and is not hidden. The mitigation, if it matters, is to
require agreement from *k of n* exemplars rather than the max — not built,
because the recall win on the case that needs it is worth more.

Reject is a hard blacklist per object and is verified to stick on re-query.

## Two of the same mug

**Measured, not hand-waved.** The `easy` suite contains instance `T`: a second
physical copy of `A`, same mould and same print run, therefore pixel-identical.
Result: **64/64 accepted as the query object, 100%.** And the timeline check
shows the consequence bluntly — `last_seen` came back as `T_07.jpg`, a photo of
the *other* copy.

This is correct behaviour for this layer and cannot be fixed inside it. Two
identical manufactured items are identical *in the pixels*; there is no signal
to separate them. Any system claiming otherwise is guessing.

What Same does about it:

1. **Never silently merges them into one confident answer.** There is one real
   piece of evidence available — if two non-overlapping, independently verified
   detections appear in a *single photo*, the user demonstrably owns at least
   two. `verify.verify_all()` finds this by re-running RANSAC on the
   correspondences left over after the first fit, and `duplicate_warning()`
   turns it into: *"2 of these appear together in IMG_4471.jpg. You own more
   than one; matches below may be different copies, and 'last seen' is the last
   time ANY copy was photographed."* `query_detailed()` returns it; the UI
   should show it above the timeline.
2. **Where it fails:** copies that never share a frame are undetectable, and the
   product will present them as one object with one misleading "last seen". Same
   does not guess — it stays silent rather than inventing a split.
3. **What would actually fix it** is out of scope for a visual matcher: wear
   marks, serial numbers, or the user splitting the entity by hand. A
   "these are two different ones" control in the UI, partitioning an object's
   exemplars, is the cheapest honest version.

The practical read: Same is reliable for objects that are *individuated* — a
scuffed teddy bear, a passport wallet, a watch with its own scratches — and
degrades to set-level answers for anything mass-produced and unmarked.

## Known limitations

* **Numbers come from synthetic suites, not a real photo library.** They are
  generated so ground truth is exact, and they cover the axes that matter
  (instance vs class, object scale, texture, blur, occlusion, duplicates). They
  do **not** cover 3D viewpoint change, lighting change, or real sensor noise.
  Treat "same-class acceptance 0.000" as strong evidence the mechanism works and
  weak evidence about absolute recall on your shelf.
  The first synthetic set was accidentally degenerate in the other direction —
  flat pixel noise, which SIFT's scale-space smooths away to ~7 keypoints — so
  this cuts both ways. See `evalset.py`.
* **`sample_photos/` (the scaffold's demo set) uses that flat-noise texture:**
  26–52 SIFT keypoints per image, giving ~0.39 recall and 5 same-class false
  positives. It under-sells the pipeline. Regenerate a better demo set with
  `python -m backend.app.retrieval.evalset easy sample_photos`.
* **Rigid 2D transform only.** An object photographed from a genuinely different
  side will not verify. Confirming it once fixes that for good — that is the
  main reason feedback exists.
* **Stage 2 is the cost.** ~25 ms per candidate on first sight, then cached in
  `.index/sift/`. `top_k=200` is ~2 s warm. Lower it for interactivity.
