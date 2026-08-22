"""Glue between the API layer and the indexing/retrieval packages.

routes.py calls only these functions — never `backend.app.indexing.*` or
`backend.app.retrieval.*` directly. Signatures below are the real, documented
public API (see backend/app/indexing/__init__.py's module docstring and
backend/app/retrieval/search.py's Searcher class).

`index_folder` is a blocking call, not an async generator, so /api/index runs
it on a background thread here and stashes progress snapshots in `_JOBS`,
keyed by job_id — that's what /api/index/{job_id}/stream polls.
"""

from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path

from backend.app.indexing import IndexStore, default_index_dir, index_folder
from backend.app.schemas import BoxXYXY, IndexProgress, PhotoRecord, QueryResponse

try:
    from backend.app.retrieval.search import Searcher
except ImportError:
    Searcher = None

_JOBS: dict[str, IndexProgress] = {}
_JOBS_LOCK = threading.Lock()

# One Searcher for the process. Constructing one loads dinov2-base (~2 s) and
# memmaps the index; doing that per request made every query pay the model load
# twice over. Reset when the index changes underneath us.
_SEARCHER = None
_SEARCHER_LOCK = threading.Lock()


def searcher():
    global _SEARCHER
    with _SEARCHER_LOCK:
        if _SEARCHER is None:
            _SEARCHER = Searcher(default_index_dir())
        return _SEARCHER


def reset_caches() -> None:
    """After an index run the store's memmaps and photo table are stale."""
    global _SEARCHER
    with _SEARCHER_LOCK:
        _SEARCHER = None


def retrieval_available() -> bool:
    return Searcher is not None


def index_exists() -> bool:
    return (default_index_dir() / "meta.json").exists()


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def start_index_job(folder_path: str, video_frame_interval_sec: float) -> str:
    job_id = uuid.uuid4().hex

    def on_progress(p: IndexProgress) -> None:
        with _JOBS_LOCK:
            _JOBS[job_id] = p

    def run() -> None:
        try:
            index_folder(
                folder_path,
                video_frame_interval_sec=video_frame_interval_sec,
                progress=on_progress,
                job_id=job_id,
            )
            reset_caches()
        except Exception as exc:  # progress callback already fired stage=error;
            # this guards the case where index_folder raised before its own
            # try/except could emit (e.g. NotADirectoryError at the top).
            with _JOBS_LOCK:
                if job_id not in _JOBS or _JOBS[job_id].stage != "error":
                    from backend.app.schemas import IndexStage

                    _JOBS[job_id] = IndexProgress(job_id=job_id, stage=IndexStage.error, error=str(exc))

    threading.Thread(target=run, daemon=True).start()
    return job_id


def get_job_progress(job_id: str) -> IndexProgress | None:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------


def indexed_folder() -> str | None:
    """The folder the user pointed us at, recovered from the indexed paths.

    Nothing writes it down, and it doesn't need to be: the common parent of
    every indexed file IS the folder. One line beats a new state file.
    """
    store = IndexStore()
    # A video frame lives in .index/frames/; the folder the USER pointed at is
    # the one holding the source video.
    paths = [e.source_video_path or e.path for e in store.photos.values()]
    if not paths:
        return None
    dirs = {str(Path(p).parent) for p in paths}
    if len(dirs) == 1:
        return dirs.pop()
    root = os.path.commonpath(list(dirs))
    # The index can hold several folders. Say so rather than reporting "/".
    return root if root not in ("/", "") else f"{len(dirs)} folders"


def photo_count() -> int:
    return len(IndexStore().photos)


def list_photos(offset: int, limit: int) -> tuple[list[PhotoRecord], int]:
    store = IndexStore()
    records = sorted(store.all_records(), key=lambda p: p.id)
    return records[offset : offset + limit], len(records)


def get_photo(photo_id: str) -> PhotoRecord | None:
    entry = IndexStore().get(photo_id)
    return entry.record if entry else None


def get_photo_path(photo_id: str) -> Path | None:
    entry = IndexStore().get(photo_id)
    return Path(entry.path) if entry else None


# ---------------------------------------------------------------------------
# Query / feedback
# ---------------------------------------------------------------------------


def run_query(
    source_photo_id: str, box: BoxXYXY, top_k: int, object_id: str | None = None
) -> QueryResponse:
    return searcher().query(source_photo_id, box, top_k=top_k, object_id=object_id)


def submit_feedback(
    query_id: str, photo_id: str, confirmed: bool, box: BoxXYXY | None = None
) -> int:
    return searcher().feedback(query_id, photo_id, confirmed, box=box)
