"""DINOv2 region descriptors + SIFT local features.

Stage 1 (recall) descriptors and stage 2 (verification) features both live here
so there is exactly one definition of "how an image becomes numbers".

WHY PATCH TOKENS, NOT CLS
    The CLS token of any vision backbone is a *scene* summary. Pooling it means
    a query crop must dominate the whole frame to match. DINOv2's patch tokens
    are instance-discriminative on their own, so we pool them over sub-windows
    and index those: a mug on a cluttered desk gets its own descriptor
    independent of the rest of the desk.

REGION LAYOUT (see region_layout())
    The image is resized to IMAGE_SIZE and run through dinov2-base (patch 14),
    giving a GRID x GRID map of patch tokens. We pool:
      * 1 whole-image region,
      * 9 half-size regions   (16x16 patches, stride 8  -> 50% overlap),
      * 25 quarter-size regions (8x8 patches, stride 6  -> ~25% overlap).
    35 descriptors per image. Overlap matters: an object straddling a hard grid
    boundary would otherwise never be well covered by any single region.

POOLING
    L2-normalize each patch token, average over the window, L2-normalize again.
    Averaging unit vectors (rather than raw tokens) stops a few high-norm
    background tokens from dominating a region that is mostly object.
"""

from __future__ import annotations

import functools
import os

# OFFLINE BY DEFAULT. `from_pretrained` contacts huggingface.co on every single
# load -- even with the weights already cached -- to check for a newer revision.
# For a product whose whole claim is "nothing leaves your Mac", a hub request on
# every query is the claim being false. Set SAME_ALLOW_MODEL_DOWNLOAD=1 for the
# one-time fetch; after that it never touches the network again.
ALLOW_MODEL_DOWNLOAD = os.environ.get("SAME_ALLOW_MODEL_DOWNLOAD") == "1"
if not ALLOW_MODEL_DOWNLOAD:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
import torch
from PIL import Image

MODEL_ID = "facebook/dinov2-base"
PATCH = 14
GRID = 32                      # 32 x 32 patch tokens
IMAGE_SIZE = GRID * PATCH      # 448
DIM = 768                      # dinov2-base hidden size

# (window_size_in_patches, stride_in_patches)
REGION_SCALES: tuple[tuple[int, int], ...] = ((16, 8), (8, 6))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@functools.lru_cache(maxsize=1)
def region_layout() -> np.ndarray:
    """(R, 4) patch-grid windows as (r0, c0, r1, c1), r1/c1 exclusive."""
    wins = [(0, 0, GRID, GRID)]
    for win, stride in REGION_SCALES:
        starts = list(range(0, GRID - win + 1, stride))
        for r0 in starts:
            for c0 in starts:
                wins.append((r0, c0, r0 + win, c0 + win))
    return np.asarray(wins, dtype=np.int32)


@functools.lru_cache(maxsize=1)
def region_boxes() -> np.ndarray:
    """(R, 4) normalized xyxy boxes for each region (schemas.BoxXYXY convention)."""
    g = float(GRID)
    lay = region_layout().astype(np.float32)
    # (r0, c0, r1, c1) -> (x1, y1, x2, y2)
    return np.stack([lay[:, 1] / g, lay[:, 0] / g, lay[:, 3] / g, lay[:, 2] / g], axis=1)


N_REGIONS = len(region_layout())


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_tensor(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1)


class Embedder:
    """Wraps dinov2-base. Weights are fetched from HuggingFace ONCE (run with
    SAME_ALLOW_MODEL_DOWNLOAD=1), then loaded strictly from the local cache --
    `local_files_only` here is what makes "no network at inference" true rather
    than merely intended."""

    def __init__(self, device: torch.device | None = None) -> None:
        from transformers import AutoModel

        self.device = device or pick_device()
        try:
            self.model = (
                AutoModel.from_pretrained(MODEL_ID, local_files_only=not ALLOW_MODEL_DOWNLOAD)
                .to(self.device)
                .eval()
            )
        except OSError as exc:
            raise RuntimeError(
                f"{MODEL_ID} is not in the local HuggingFace cache. Same never downloads "
                "during normal use. Run the one-time fetch:\n"
                "    SAME_ALLOW_MODEL_DOWNLOAD=1 uv run python -c "
                "'from backend.app.indexing import Embedder; Embedder()'"
            ) from exc

    @torch.inference_mode()
    def _patch_tokens(self, imgs: list[Image.Image]) -> torch.Tensor:
        """(B, GRID, GRID, DIM), each token L2-normalized."""
        x = torch.stack([_to_tensor(i) for i in imgs]).to(self.device)
        out = self.model(pixel_values=x).last_hidden_state  # (B, 1 + GRID*GRID, DIM)
        tok = out[:, 1:, :]
        tok = torch.nn.functional.normalize(tok, dim=-1)
        return tok.reshape(len(imgs), GRID, GRID, DIM)

    @torch.inference_mode()
    def embed_images(self, imgs: list[Image.Image]) -> np.ndarray:
        """(B, N_REGIONS, DIM) float32, L2-normalized region descriptors."""
        tok = self._patch_tokens(imgs)
        # integral image over the patch grid -> O(1) mean pooling per window
        cum = tok.cumsum(1).cumsum(2)
        cum = torch.nn.functional.pad(cum, (0, 0, 1, 0, 1, 0))  # pad row0/col0 with zeros
        lay = region_layout()
        r0, c0, r1, c1 = (torch.as_tensor(lay[:, i], device=tok.device) for i in range(4))
        total = cum[:, r1, c1] - cum[:, r0, c1] - cum[:, r1, c0] + cum[:, r0, c0]
        counts = ((r1 - r0) * (c1 - c0)).to(total.dtype).view(1, -1, 1)
        regions = torch.nn.functional.normalize(total / counts, dim=-1)
        return regions.float().cpu().numpy()

    @torch.inference_mode()
    def embed_crops(self, img: Image.Image, boxes: np.ndarray) -> np.ndarray:
        """(K, DIM) whole-crop descriptors for normalized xyxy `boxes` of `img`.

        Used for the query side: the user's box becomes its own image, so a tiny
        object gets the backbone's full resolution instead of a handful of
        patches. Feed several dilated boxes to hedge against scale mismatch with
        the indexed regions.
        """
        w, h = img.size
        crops = []
        for x1, y1, x2, y2 in np.asarray(boxes, dtype=np.float32).reshape(-1, 4):
            px = (
                int(np.clip(x1, 0, 1) * w),
                int(np.clip(y1, 0, 1) * h),
                max(int(np.clip(x2, 0, 1) * w), int(np.clip(x1, 0, 1) * w) + 1),
                max(int(np.clip(y2, 0, 1) * h), int(np.clip(y1, 0, 1) * h) + 1),
            )
            crops.append(img.convert("RGB").crop(px))
        return self.embed_images(crops)[:, 0, :]  # region 0 == whole image


# --------------------------------------------------------------------------
# Stage 2: SIFT local features for geometric verification
# --------------------------------------------------------------------------

SIFT_MAX_SIDE = 1024      # cap cost; keypoints are mapped back to full-res coords
SIFT_MAX_FEATURES = 3000


@functools.lru_cache(maxsize=1)
def _sift() -> "cv2.SIFT":
    return cv2.SIFT_create(nfeatures=SIFT_MAX_FEATURES)


def sift_features(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """SIFT keypoints + descriptors.

    Returns (xy (N,2) float32, desc (N,128) float32). Keypoint coordinates are
    in PIXELS OF `img` AS PASSED IN (the detector runs on a downscaled copy but
    coordinates are scaled back), so a RANSAC reprojection threshold in pixels
    means what you expect.
    """
    arr = np.asarray(img.convert("L"))
    h, w = arr.shape
    scale = min(1.0, SIFT_MAX_SIDE / max(h, w))
    if scale < 1.0:
        arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    kp, desc = _sift().detectAndCompute(arr, None)
    if not kp:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    xy = np.array([k.pt for k in kp], dtype=np.float32) / scale
    return xy, desc.astype(np.float32)
