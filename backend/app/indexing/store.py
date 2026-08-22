"""On-disk index. Everything lives under `.index/` and nothing else is needed
to serve queries after a restart.

LAYOUT
    .index/meta.json        index version, model + geometry config, row count
    .index/photos.json      photo_id -> PhotoEntry (incl. region_start/count)
    .index/regions.f16      raw float16 (N_rows, DIM), append-only
    .index/boxes.f32        raw float32 (N_rows, 4), normalized xyxy, append-only
    .index/hnsw.bin         hnswlib index; label == row id into the two raw files
    .index/frames/          JPEGs sampled out of video files
    .index/sift/<id>.npz    lazily-filled SIFT cache (see sift_for())

WHY RAW APPEND-ONLY FILES
    Adding photos on a re-run must not rewrite gigabytes. `.npy` cannot be
    appended to without rewriting its header/body, so the two big arrays are
    plain little-endian dumps with the shape recorded in meta.json and read back
    with np.memmap. Deleting or modifying a photo leaves its rows as holes
    (marked deleted in hnswlib, no longer referenced by photos.json). Holes are
    never reclaimed; `rm -rf .index` and re-index if a library churns enough for
    that to matter.

DISK / TIME TRADEOFF FOR STAGE 2
    SIFT descriptors cost ~300 KB/image (3000 x 128 float32) - roughly 4x the
    54 KB of DINOv2 region descriptors - but stage 2 only ever touches the
    shortlist (<= top_k images per query), so precomputing them for a whole
    library is mostly wasted disk. They are computed on demand and cached per
    photo, so the first query on an object pays ~25 ms/candidate and every
    later one pays a file read.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from ..schemas import BoxXYXY, GpsCoord, PhotoRecord
from . import features as F

SIFT_MEM = 4096          # photos whose SIFT stays resident in RAM

INDEX_VERSION = 1
_HNSW_M = 32
_HNSW_EF_CONSTRUCTION = 200
_HNSW_EF_SEARCH = 128


def default_index_dir() -> Path:
    return Path(os.environ.get("SAME_INDEX_DIR", Path.cwd() / ".index"))


@dataclass
class PhotoEntry:
    """One indexed image or sampled video frame, plus the bookkeeping the
    indexer needs that `PhotoRecord` (the API contract) does not carry."""

    id: str
    path: str
    width: int
    height: int
    mtime_ns: int
    size: int
    region_start: int = 0
    region_count: int = 0
    taken_at: str | None = None
    taken_at_inferred: bool = True   # True => derived from file mtime, not EXIF
    gps: list[float] | None = None
    source_video_path: str | None = None
    frame_time_sec: float | None = None

    @property
    def record(self) -> PhotoRecord:
        return PhotoRecord(
            id=self.id,
            path=self.path,
            source_video_path=self.source_video_path,
            frame_time_sec=self.frame_time_sec,
            taken_at=datetime.fromisoformat(self.taken_at) if self.taken_at else None,
            gps=GpsCoord(lat=self.gps[0], lon=self.gps[1]) if self.gps else None,
            width=self.width,
            height=self.height,
        )


@dataclass
class RegionHit:
    """A stage-1 recall hit: one indexed region that looked like the query."""

    photo_id: str
    score: float                 # cosine similarity in [-1, 1], practically [0, 1]
    box: BoxXYXY                 # where in the candidate photo, normalized xyxy
    row: int                     # row id in regions.f16, for debugging


class IndexStore:
    """Open (or create) the index at `root`. Cheap: the big arrays are memmapped
    and the hnswlib graph is loaded once, lazily, on first search."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else default_index_dir()
        self.frames_dir = self.root / "frames"
        self.sift_dir = self.root / "sift"
        self.meta: dict = {"version": INDEX_VERSION, "rows": 0}
        self.photos: dict[str, PhotoEntry] = {}
        self._hnsw = None
        self._regions: np.memmap | None = None
        self._boxes: np.memmap | None = None
        self._row_owner: np.ndarray | None = None
        self.photo_ids: list[str] = []
        self._sift_mem: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._sift_lock = threading.Lock()
        self.load()

    # -- persistence ------------------------------------------------------
    @property
    def rows(self) -> int:
        return int(self.meta.get("rows", 0))

    def load(self) -> None:
        mp, pp = self.root / "meta.json", self.root / "photos.json"
        if mp.exists():
            self.meta = json.loads(mp.read_text())
            if self.meta.get("version") != INDEX_VERSION or self.meta.get("model") != F.MODEL_ID:
                raise RuntimeError(
                    f"{self.root} was built by a different index version/model "
                    f"({self.meta.get('version')}/{self.meta.get('model')}). Delete it and re-index."
                )
        if pp.exists():
            self.photos = {k: PhotoEntry(**v) for k, v in json.loads(pp.read_text()).items()}

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta.update(
            version=INDEX_VERSION, model=F.MODEL_ID, dim=F.DIM,
            image_size=F.IMAGE_SIZE, grid=F.GRID, regions_per_image=F.N_REGIONS,
            photos=len(self.photos),
        )
        (self.root / "meta.json").write_text(json.dumps(self.meta, indent=2))
        (self.root / "photos.json").write_text(
            json.dumps({k: asdict(v) for k, v in self.photos.items()})
        )
        if self._hnsw is not None:
            self._hnsw.save_index(str(self.root / "hnsw.bin"))
        self._regions = self._boxes = self._row_owner = None  # force remap

    # -- writing ----------------------------------------------------------
    def append_regions(self, vecs: np.ndarray, boxes: np.ndarray) -> int:
        """Append (R, DIM) descriptors + (R, 4) boxes. Returns the first row id."""
        self.root.mkdir(parents=True, exist_ok=True)
        start = self.rows
        with open(self.root / "regions.f16", "ab") as fh:
            fh.write(np.ascontiguousarray(vecs, dtype=np.float16).tobytes())
        with open(self.root / "boxes.f32", "ab") as fh:
            fh.write(np.ascontiguousarray(boxes, dtype=np.float32).tobytes())
        self.meta["rows"] = start + len(vecs)
        self._regions = self._boxes = self._row_owner = None
        self._hnsw_add(np.arange(start, start + len(vecs)), vecs)
        return start

    def drop_photo(self, photo_id: str) -> None:
        """Forget a photo. Its rows become holes (see module docstring)."""
        entry = self.photos.pop(photo_id, None)
        if entry is None:
            return
        idx = self._hnsw_open()
        for row in range(entry.region_start, entry.region_start + entry.region_count):
            try:
                idx.mark_deleted(row)
            except RuntimeError:
                pass  # already deleted
        (self.sift_dir / f"{photo_id}.npz").unlink(missing_ok=True)
        self._row_owner = None

    # -- reading ----------------------------------------------------------
    def _mm(self, name: str, dtype, cols: int) -> np.ndarray:
        path = self.root / name
        if self.rows == 0 or not path.exists():
            return np.zeros((0, cols), dtype=dtype)
        return np.memmap(path, dtype=dtype, mode="r", shape=(self.rows, cols))

    @property
    def regions(self) -> np.ndarray:
        if self._regions is None:
            self._regions = self._mm("regions.f16", np.float16, F.DIM)
        return self._regions

    @property
    def boxes(self) -> np.ndarray:
        if self._boxes is None:
            self._boxes = self._mm("boxes.f32", np.float32, 4)
        return self._boxes

    @property
    def row_owner(self) -> np.ndarray:
        """(rows,) int32 index into `self.photo_ids`; -1 for holes."""
        if self._row_owner is None:
            owner = np.full(self.rows, -1, dtype=np.int32)
            self.photo_ids = list(self.photos)
            for i, pid in enumerate(self.photo_ids):
                e = self.photos[pid]
                owner[e.region_start : e.region_start + e.region_count] = i
            self._row_owner = owner
        return self._row_owner

    def get(self, photo_id: str) -> PhotoEntry | None:
        return self.photos.get(photo_id)

    def all_records(self) -> list[PhotoRecord]:
        return [e.record for e in self.photos.values()]

    # -- hnswlib ----------------------------------------------------------
    def _hnsw_open(self):
        if self._hnsw is None:
            import hnswlib

            idx = hnswlib.Index(space="ip", dim=F.DIM)  # vectors are unit norm => ip == cosine
            path = self.root / "hnsw.bin"
            if path.exists():
                idx.load_index(str(path), max_elements=max(self.rows, 1))
            else:
                idx.init_index(
                    max_elements=max(self.rows, 1024),
                    M=_HNSW_M,
                    ef_construction=_HNSW_EF_CONSTRUCTION,
                )
            idx.set_ef(_HNSW_EF_SEARCH)
            self._hnsw = idx
        return self._hnsw

    def _hnsw_add(self, labels: np.ndarray, vecs: np.ndarray) -> None:
        idx = self._hnsw_open()
        need = int(labels.max()) + 1
        if need > idx.get_max_elements():
            idx.resize_index(max(need, int(idx.get_max_elements() * 2)))
        idx.add_items(np.ascontiguousarray(vecs, dtype=np.float32), labels)

    def search(self, query: np.ndarray, top_k_photos: int = 50,
               regions_per_photo: int = 20) -> list[RegionHit]:
        """Stage 1 recall. `query` is (DIM,) or (K, DIM) unit-norm descriptors;
        with several the best-scoring one per photo wins (query-side multi-scale).

        Returns at most `top_k_photos` hits, one per photo, best region first.
        These are CANDIDATES — they have had no geometric verification and must
        not be shown to a user as matches.
        """
        q = np.atleast_2d(np.asarray(query, dtype=np.float32))
        if self.rows == 0 or not self.photos:
            return []
        idx = self._hnsw_open()
        alive = int((self.row_owner >= 0).sum())
        k = min(max(top_k_photos * regions_per_photo, top_k_photos), alive)
        if k == 0:
            return []
        idx.set_ef(max(_HNSW_EF_SEARCH, k))  # hnswlib needs ef >= k or recall silently drops
        labels, dists = idx.knn_query(q, k=k)
        best: dict[str, RegionHit] = {}
        for row, dist in zip(labels.ravel(), dists.ravel()):
            owner = self.row_owner[row]
            if owner < 0:
                continue
            pid = self.photo_ids[owner]
            score = float(1.0 - dist)
            prev = best.get(pid)
            if prev is None or score > prev.score:
                x1, y1, x2, y2 = (float(v) for v in self.boxes[row])
                best[pid] = RegionHit(
                    photo_id=pid, score=score,
                    box=BoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2), row=int(row),
                )
        return sorted(best.values(), key=lambda h: -h.score)[:top_k_photos]

    # -- stage-2 feature cache -------------------------------------------
    def sift_for(self, photo_id: str) -> tuple[np.ndarray, np.ndarray]:
        """SIFT keypoints/descriptors for an indexed photo, computed on first
        request and cached to `.index/sift/`. Coordinates are pixels in the
        photo's own full-resolution frame."""
        entry = self.photos.get(photo_id)
        if entry is None:
            raise KeyError(photo_id)
        hot = self._sift_mem.get(photo_id)
        if hot is not None:
            return hot
        cache = self.sift_dir / f"{photo_id}.npz"
        if cache.exists():
            with np.load(cache) as z:
                out = z["xy"], z["desc"].astype(np.float32)
            self._remember_sift(photo_id, out)
            return out
        xy, desc = F.sift_features(open_image(entry.path))
        self.sift_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache, xy=xy, desc=desc.astype(np.float16))
        self._remember_sift(photo_id, (xy, desc))
        return xy, desc

    # A query verifies every shortlisted photo, so the same descriptors are read
    # over and over. Keeping the last SIFT_MEM of them resident turns the second
    # query over a library into pure compute. ~1.5 MB per 12 MP photo.
    def _remember_sift(self, photo_id: str, value) -> None:
        with self._sift_lock:
            if len(self._sift_mem) >= SIFT_MEM:
                self._sift_mem.pop(next(iter(self._sift_mem)))
            self._sift_mem[photo_id] = value


def open_image(path: str | Path) -> Image.Image:
    """Load an image with EXIF orientation already applied.

    Browsers auto-rotate JPEGs by their EXIF orientation tag, so the indexer
    must too — otherwise a normalized box drawn in the UI lands somewhere else
    in the array we embedded.
    """
    from PIL import ImageOps

    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")
