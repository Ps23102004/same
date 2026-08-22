"""Stage 2: geometric verification. This is the step that makes the product real.

Stage 1 answers "which photos contain something that LOOKS like this". On its
own that is a category retriever, and a category retriever is a failed product:
ask for your teddy bear and you get other people's teddy bears.

Stage 2 demands more than appearance. It matches SIFT descriptors between the
query crop and each candidate, then asks RANSAC whether those correspondences
agree on ONE rigid 2D transform. Two different teddy bears throw off plenty of
ratio-test matches -- same silhouette, same fur statistics -- but the matches
land in geometrically incoherent places, so a consensus set never forms and the
inlier count collapses to single digits. The same bear photographed twice keeps
a coherent set. That gap is the whole discriminator, and the inlier count is
reported to the UI as the confidence rather than being laundered into a
percentage.

Feature extraction lives in indexing.features.sift_features; this module is
purely the matching + consensus half.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from backend.app.indexing import features as F

RATIO = 0.85              # Lowe's ratio test, tuned in eval.py (see below)
RANSAC_REPROJ_PX = 5.0
MIN_MATCHES = 4           # fewer than this and no transform can be estimated at all
QUERY_MIN_SIDE = 256      # upscale tiny crops; SIFT needs pixels to find anything
MIN_QUERY_KEYPOINTS = 40  # below this, the crop really is too small -- then upscale

# RATIO was swept on the eval suites. 0.75 -> 0.85 raises recall on the hard
# (flat-surface, blurred, occluded) suite from 0.661 to 0.732 at unchanged
# precision; it also lifts lookalike inlier counts, which is exactly why the
# accept rule below leans on inlier_fraction rather than a raw count.
# One matcher PER THREAD. Candidates are verified in parallel (search.py), and
# a single cv2 matcher object shared across threads is not safe to reuse.
_TLS = threading.local()


def _flann() -> "cv2.FlannBasedMatcher":
    m = getattr(_TLS, "flann", None)
    if m is None:
        m = _TLS.flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
    return m

# RANSAC samples randomly. For a well-textured object with 130 correspondences
# that is irrelevant, but a flat blurred object sits at 4-6 correspondences and
# the same query can land either side of the accept gate run to run (measured:
# recall 0.21 vs 0.46 on the hard suite across two runs). Pinning the seed makes
# results reproducible; it does NOT remove the underlying fragility, which is a
# real property of the hard case and is reported as such in the README.
cv2.setRNGSeed(0)


@dataclass
class Verification:
    inliers: int                                     # THE confidence signal
    matches: int                                     # ratio-test survivors, pre-RANSAC
    box: tuple[float, float, float, float] | None    # normalized xyxy in the candidate
    scale: float = 0.0                               # candidate size / query size

    @property
    def inlier_fraction(self) -> float:
        """Share of ratio-test correspondences that agreed on one transform.

        The absolute inlier count is NOT scale-invariant: a busy patterned bag
        yields 130 inliers, a plain leather wallet yields 5, and no single
        absolute threshold separates both from lookalikes (measured -- see
        README). The fraction is: a true match has most of its correspondences
        agreeing, a lookalike has almost none.
        """
        return self.inliers / max(self.matches, 1)


def crop_features(img: Image.Image, box) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """SIFT for a user-drawn box. Returns (xy, desc, (w, h)) in crop pixels.

    THE CROP IS RENDERED AT THE SAME EFFECTIVE SCALE AS THE INDEXED IMAGES.
    Indexed photos are detected at SIFT_MAX_SIDE (1024) px, so a 3840 px frame
    is seen 3.75x smaller than life. A query crop taken at native pixels is
    therefore compared against candidate keypoints computed at a completely
    different level of detail, and the ratio test degenerates: measured on real
    4K footage, TRUE matches came back at inlier_fraction 0.24-0.38 -- below the
    0.40 accept gate, i.e. the product returned nothing for an object that was
    plainly there. Pre-scaling the crop by the same factor moved the same
    matches to 0.62-0.72. For images at or below 1024 px nothing changes.

    Only if that leaves too few keypoints do we fall back to upscaling, which is
    what rescues a genuinely tiny crop out of a small image.
    """
    w, h = img.size
    x1, y1, x2, y2 = (float(v) for v in box)
    px = (
        int(np.clip(x1, 0, 1) * w),
        int(np.clip(y1, 0, 1) * h),
        max(int(np.clip(x2, 0, 1) * w), int(np.clip(x1, 0, 1) * w) + 2),
        max(int(np.clip(y2, 0, 1) * h), int(np.clip(y1, 0, 1) * h) + 2),
    )
    crop = img.convert("RGB").crop(px)

    index_scale = min(1.0, F.SIFT_MAX_SIDE / max(w, h))
    if index_scale < 1.0:
        crop = crop.resize(
            (max(2, int(crop.width * index_scale)), max(2, int(crop.height * index_scale))),
            Image.LANCZOS,
        )
    xy, desc = F.sift_features(crop)

    s = max(1.0, QUERY_MIN_SIDE / max(1, min(crop.size)))
    if s > 1.0 and len(desc) < MIN_QUERY_KEYPOINTS:
        crop = crop.resize((int(crop.width * s), int(crop.height * s)), Image.BICUBIC)
        xy, desc = F.sift_features(crop)
    return xy, desc, crop.size


def _fit(src, dst, full_affine):
    est = cv2.estimateAffine2D if full_affine else cv2.estimateAffinePartial2D
    return est(src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ_PX)


def _project(M, q_size, c_size):
    qw, qh = q_size
    cw, ch = c_size
    proj = cv2.transform(
        np.float32([[0, 0], [qw, 0], [qw, qh], [0, qh]]).reshape(-1, 1, 2), M
    ).reshape(-1, 2)
    x1, y1 = proj.min(0)
    x2, y2 = proj.max(0)
    return (
        float(np.clip(x1 / cw, 0, 1)), float(np.clip(y1 / ch, 0, 1)),
        float(np.clip(x2 / cw, 0, 1)), float(np.clip(y2 / ch, 0, 1)),
    )


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def verify_all(
    q_xy: np.ndarray,
    q_desc: np.ndarray,
    q_size: tuple[int, int],
    c_xy: np.ndarray,
    c_desc: np.ndarray,
    c_size: tuple[int, int],
    *,
    full_affine: bool = False,
    max_detections: int = 2,
) -> list[Verification]:
    """Every geometrically consistent detection of the query in the candidate.

    Usually one. TWO non-overlapping detections in a single photo is the one
    piece of hard evidence the pixels can give us that the user owns more than
    one of this item -- two of the same mug on the same table. It cannot catch
    duplicates that never share a frame, but when it fires it is proof, and the
    UI can stop claiming "your mug was last seen here" as if there were one.

    Implemented by repeated RANSAC: fit, take the consensus set out, fit again
    on what is left. Sorted by inlier count, best first.
    """
    if q_desc is None or c_desc is None or len(q_desc) < 2 or len(c_desc) < 2:
        return []
    pairs = _flann().knnMatch(np.ascontiguousarray(q_desc, np.float32),
                            np.ascontiguousarray(c_desc, np.float32), k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    n_good = len(good)
    if n_good < MIN_MATCHES:
        return [Verification(0, n_good, None)]

    dets: list[Verification] = []
    while len(good) >= MIN_MATCHES and len(dets) < max_detections:
        src = np.float32([q_xy[m.queryIdx] for m in good]).reshape(-1, 1, 2)
        dst = np.float32([c_xy[m.trainIdx] for m in good]).reshape(-1, 1, 2)
        M, mask = _fit(src, dst, full_affine)
        if M is None or mask is None:
            break
        keep = mask.ravel().astype(bool)
        inliers = int(keep.sum())
        if inliers < MIN_MATCHES:
            break
        box = _project(M, q_size, c_size)
        # A transform that maps the crop onto a sliver is a degenerate fit.
        if (box[2] - box[0]) >= 0.01 and (box[3] - box[1]) >= 0.01:
            if all(_iou(box, d.box) < 0.3 for d in dets):
                dets.append(
                    Verification(inliers, n_good, box, float(np.hypot(M[0, 0], M[0, 1])))
                )
        good = [m for m, k in zip(good, keep) if not k]
    return dets or [Verification(0, n_good, None)]


def verify(
    q_xy: np.ndarray,
    q_desc: np.ndarray,
    q_size: tuple[int, int],
    c_xy: np.ndarray,
    c_desc: np.ndarray,
    c_size: tuple[int, int],
    *,
    full_affine: bool = False,
) -> Verification:
    """RANSAC a transform from query-crop pixels to candidate pixels.

    `full_affine=False` fits 4 DoF (rotation + uniform scale + translation).
    That is the most constrained model and therefore the most discriminative,
    which is what we want when the job is rejecting a same-class-different-
    instance lookalike. `full_affine=True` (6 DoF) is the escape hatch for flat
    objects shot at strong angles; a full homography is the next rung up and
    accepts correspondingly more junk.
    """
    return verify_all(q_xy, q_desc, q_size, c_xy, c_desc, c_size,
                      full_affine=full_affine, max_detections=1)[0]
