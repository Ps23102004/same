"""Walk a folder, sample video frames, read EXIF, embed, persist. Incremental.

Everything here is local: ffmpeg on disk, dinov2-base from the transformers
cache, no network at index or query time.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from ..schemas import IndexProgress, IndexStage
from . import features as F
from .store import IndexStore, PhotoEntry, open_image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
BATCH_SIZE = 8

ProgressCb = Callable[[IndexProgress], None]


@dataclass
class IndexStats:
    photos_indexed: int = 0     # newly embedded this run
    photos_skipped: int = 0     # unchanged, left alone
    photos_removed: int = 0     # gone from disk or changed
    frames_extracted: int = 0
    total_photos: int = 0       # size of the index afterwards
    seconds: float = 0.0
    index_bytes: int = 0

    @property
    def images_per_sec(self) -> float:
        return self.photos_indexed / self.seconds if self.seconds else 0.0


# ---------------------------------------------------------------------------
# EXIF
# ---------------------------------------------------------------------------

def _dms(v) -> float:
    d, m, s = (float(x) for x in v)
    return d + m / 60.0 + s / 3600.0


def read_exif(img: Image.Image) -> tuple[datetime | None, tuple[float, float] | None]:
    """(capture time, (lat, lon)). Either may be None; garbage never raises."""
    taken = gps = None
    try:
        exif = img.getexif()
    except Exception:
        return None, None
    # DateTimeOriginal/DateTimeDigitized live in the Exif SUB-IFD (0x8769), which
    # Image.getexif() does not merge into the top level — only DateTime (0x0132)
    # is in the 0th IFD. Reading just the top level silently loses the real
    # capture time on almost every camera JPEG.
    try:
        sub = exif.get_ifd(0x8769)
    except Exception:
        sub = {}
    for src, tag in ((sub, 0x9003), (sub, 0x9004), (exif, 0x0132)):
        raw = src.get(tag) if src else None
        if not raw:
            continue
        try:
            taken = datetime.strptime(str(raw).strip()[:19], "%Y:%m:%d %H:%M:%S")
            break
        except (ValueError, TypeError):
            continue  # phones and editors write plenty of malformed dates
    try:
        g = exif.get_ifd(0x8825)
        if g and 2 in g and 4 in g:
            lat, lon = _dms(g[2]), _dms(g[4])
            if str(g.get(1, "N")).upper().startswith("S"):
                lat = -lat
            if str(g.get(3, "E")).upper().startswith("W"):
                lon = -lon
            if -90 <= lat <= 90 and -180 <= lon <= 180 and (lat or lon):
                gps = (lat, lon)
    except Exception:
        pass
    return taken, gps


# ---------------------------------------------------------------------------
# Video frame sampling
# ---------------------------------------------------------------------------

def _video_creation_time(path: Path) -> datetime | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
            capture_output=True, text=True, timeout=30,
        ).stdout
        raw = json.loads(out)["format"]["tags"]["creation_time"]
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        return None


def sample_video(video: Path, out_dir: Path, interval_sec: float) -> list[tuple[Path, float]]:
    """Extract one frame every `interval_sec` seconds. Returns [(frame_path, t)].

    Frame times are nominal (i * interval); ffmpeg's fps filter picks the nearest
    real frame, so they can drift by up to one source frame. Fine for "when did
    I last see this" — not a timecode.
    """
    stem = hashlib.sha1(str(video.resolve()).encode()).hexdigest()[:16]
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{stem}_*.jpg"):
        old.unlink()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps=1/{interval_sec}",
         "-qscale:v", "3", str(out_dir / f"{stem}_%05d.jpg")],
        check=True, capture_output=True, timeout=3600,
    )
    frames = sorted(out_dir.glob(f"{stem}_*.jpg"))
    return [(p, i * interval_sec) for i, p in enumerate(frames)]


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def _photo_id(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()[:16]


@dataclass
class _Pending:
    """A file (or extracted frame) that needs embedding."""
    id: str
    file: Path            # what to actually open and embed
    mtime_ns: int         # of the ORIGINAL source, for change detection
    size: int
    source_video: Path | None = None
    frame_time: float | None = None
    video_start: datetime | None = None


def scan(folder: Path, store: IndexStore, interval_sec: float,
         report: Callable[[str], None] = lambda _: None) -> tuple[list[_Pending], set[str], int, int]:
    """Return (pending items to embed, ids to drop, frames extracted, unchanged count).

    A file is re-embedded only if its (mtime_ns, size) changed; that is the whole
    incrementality story and it costs one stat() per file.
    """
    seen: set[str] = set()
    pending: list[_Pending] = []
    frames_extracted = 0
    index_root = store.root.resolve()

    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if index_root in path.resolve().parents:
            continue  # never index our own extracted frames as source photos
        ext = path.suffix.lower()
        st = path.stat()

        if ext in IMAGE_EXTS:
            pid = _photo_id(str(path.resolve()))
            seen.add(pid)
            old = store.get(pid)
            if old and old.mtime_ns == st.st_mtime_ns and old.size == st.st_size:
                continue
            pending.append(_Pending(pid, path, st.st_mtime_ns, st.st_size))

        elif ext in VIDEO_EXTS:
            key = str(path.resolve())
            # One probe id for the video itself so we can detect "unchanged".
            existing = [e for e in store.photos.values() if e.source_video_path == key]
            if existing and all(
                e.mtime_ns == st.st_mtime_ns and e.size == st.st_size for e in existing
            ):
                seen.update(e.id for e in existing)
                continue
            report(f"sampling frames from {path.name}")
            start = _video_creation_time(path)
            try:
                frames = sample_video(path, store.frames_dir, interval_sec)
            except FileNotFoundError:
                # ffmpeg is not installed. One missing optional tool must not
                # fail the whole library the way an unreadable image doesn't
                # (see the per-file guard in the embed loop below).
                report("ffmpeg not found on PATH — skipping videos "
                       "(brew install ffmpeg to index them)")
                continue
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                report(f"could not sample {path.name}: {exc}")
                continue
            for frame_path, t in frames:
                frames_extracted += 1
                pid = _photo_id(f"{key}@{t}")
                seen.add(pid)
                pending.append(
                    _Pending(pid, frame_path, st.st_mtime_ns, st.st_size,
                             source_video=path, frame_time=t, video_start=start)
                )

    # Only photos that live UNDER this folder can be stale. Without the
    # containment test, `set(store.photos) - seen` silently deletes every photo
    # from every OTHER folder the user has indexed -- one index, many folders,
    # and indexing the second one wiped the first. (Found end to end: indexing a
    # second folder dropped photo_count from 72 to 36.)
    def _under(entry) -> bool:
        src = Path(entry.source_video_path or entry.path)
        return folder == src or folder in src.parents

    stale = {pid for pid, e in store.photos.items() if pid not in seen and _under(e)}
    return pending, stale, frames_extracted, len(seen) - len(pending)


# ---------------------------------------------------------------------------
# The indexer
# ---------------------------------------------------------------------------

def index_folder(
    folder: str | Path,
    index_dir: str | Path | None = None,
    video_frame_interval_sec: float = 2.0,
    progress: ProgressCb | None = None,
    job_id: str = "index",
    embedder: F.Embedder | None = None,
) -> IndexStats:
    """Index (or re-index) `folder` into `index_dir`. Safe to re-run: unchanged
    files are skipped, changed files re-embedded, deleted files dropped."""
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    store = IndexStore(index_dir)
    t0 = time.perf_counter()
    stats = IndexStats()

    prog = IndexProgress(job_id=job_id, stage=IndexStage.scanning)

    def emit(**kw) -> None:
        for k, v in kw.items():
            setattr(prog, k, v)
        if progress:
            progress(prog.model_copy(deep=True))

    try:
        emit(message=f"scanning {folder}")
        pending, stale, frames, unchanged = scan(
            folder, store, video_frame_interval_sec,
            report=lambda m: emit(stage=IndexStage.extracting_frames, message=m),
        )
        stats.frames_extracted = frames
        stats.photos_skipped = unchanged

        for pid in stale:
            store.drop_photo(pid)
        stats.photos_removed = len(stale)

        emit(stage=IndexStage.embedding, total_files=len(pending), processed_files=0,
             total_frames=frames, processed_frames=frames,
             message=f"{len(pending)} new/changed, {stats.photos_skipped} unchanged")

        if embedder is None and pending:
            embedder = F.Embedder()

        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i : i + BATCH_SIZE]
            imgs, entries = [], []
            for p in batch:
                try:
                    img = open_image(p.file)
                except Exception as exc:  # unreadable/corrupt file: skip, keep going
                    emit(message=f"skipped {p.file.name}: {exc}")
                    continue
                taken, gps = read_exif(img)
                # "inferred" means we GUESSED from the filesystem. A video's
                # container creation_time is real metadata, so a frame derived
                # from it is not inferred.
                inferred = taken is None and p.video_start is None
                if taken is None:
                    # No usable EXIF: fall back to the video's container
                    # creation_time, else the file's mtime, offset by where the
                    # frame sits in the video. Flagged so the UI can say "approx".
                    base = p.video_start or datetime.fromtimestamp(p.mtime_ns / 1e9)
                    taken = base + timedelta(seconds=p.frame_time or 0.0)
                imgs.append(img)
                entries.append(
                    PhotoEntry(
                        id=p.id,
                        path=str(p.file.resolve()),
                        width=img.width, height=img.height,
                        mtime_ns=p.mtime_ns, size=p.size,
                        taken_at=taken.isoformat() if taken else None,
                        taken_at_inferred=inferred,
                        gps=[gps[0], gps[1]] if gps else None,
                        source_video_path=str(p.source_video.resolve()) if p.source_video else None,
                        frame_time_sec=p.frame_time,
                    )
                )
            if not imgs:
                continue
            vecs = embedder.embed_images(imgs)          # (B, R, D)
            n_regions = vecs.shape[1]
            # One append for the whole batch: 8 photos = 1 file write + 1 hnsw
            # add_items instead of 8 of each. Rows stay contiguous per photo.
            start = store.append_regions(
                vecs.reshape(-1, vecs.shape[-1]),
                np.tile(F.region_boxes(), (len(entries), 1)),
            )
            for j, entry in enumerate(entries):
                entry.region_start = start + j * n_regions
                entry.region_count = n_regions
                store.photos[entry.id] = entry
                stats.photos_indexed += 1
            emit(processed_files=min(i + BATCH_SIZE, len(pending)))

        emit(stage=IndexStage.building_index, message="persisting index")
        store.save()
        stats.total_photos = len(store.photos)
        stats.seconds = time.perf_counter() - t0
        stats.index_bytes = sum(f.stat().st_size for f in store.root.rglob("*") if f.is_file())
        emit(stage=IndexStage.done,
             message=f"{stats.total_photos} photos, {stats.images_per_sec:.2f} img/s")
        return stats
    except Exception as exc:
        emit(stage=IndexStage.error, error=str(exc))
        raise
