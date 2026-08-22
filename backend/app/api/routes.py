"""All HTTP endpoints for Same. FastAPI router mounted by main.py."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

from backend.app.api import adapters
from backend.app.schemas import (  # noqa: F401
    FeedbackRequest,
    IndexProgress,
    IndexRequest,
    IndexStage,
    PhotoRecord,
    QueryRequest,
    QueryResponse,
)

router = APIRouter(prefix="/api")

THUMBNAIL_DIR = Path(__file__).resolve().parents[3] / ".index" / "thumbnails"
THUMBNAIL_SIZE = (320, 320)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health")
def health() -> dict:
    import torch

    return {
        "status": "ok",
        "mps_available": torch.backends.mps.is_available(),
        "retrieval_module_loaded": adapters.retrieval_available(),
        "index_exists": adapters.index_exists(),
    }


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


@router.get("/status")
def status() -> dict:
    """What the UI asks on boot: is there an index, how big, of what folder."""
    if not adapters.index_exists():
        return {"indexed": False, "photo_count": 0, "folder_path": None}
    return {
        "indexed": True,
        "photo_count": adapters.photo_count(),
        "folder_path": adapters.indexed_folder(),
    }


@router.post("/index", response_model=IndexProgress)
def start_index(req: IndexRequest) -> IndexProgress:
    if not Path(req.folder_path).is_dir():
        raise HTTPException(400, f"folder does not exist: {req.folder_path}")
    job_id = adapters.start_index_job(req.folder_path, req.video_frame_interval_sec)
    # A full IndexProgress, not a bare {job_id}: the client renders this object
    # immediately and would otherwise read counters off undefined.
    return IndexProgress(job_id=job_id, stage=IndexStage.scanning, message="starting")


@router.get("/index/{job_id}", response_model=IndexProgress)
def index_progress(job_id: str) -> IndexProgress:
    """Poll. (The SSE variant below is equivalent; the UI polls.)"""
    progress = adapters.get_job_progress(job_id)
    if progress is None:
        raise HTTPException(404, "unknown job_id")
    return progress


@router.get("/index/{job_id}/stream")
async def stream_index_progress(job_id: str):
    """SSE stream of IndexProgress until the job reaches done/error."""

    async def events():
        while True:
            progress = adapters.get_job_progress(job_id)
            if progress is None:
                yield f"event: error\ndata: {json.dumps({'error': 'unknown job_id'})}\n\n"
                return
            yield f"data: {progress.model_dump_json()}\n\n"
            if progress.stage in (IndexStage.done, IndexStage.error):
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(events(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------


@router.get("/photos")
def list_photos(offset: int = 0, limit: int = 50) -> dict:
    if not adapters.index_exists():
        raise HTTPException(409, "no index yet — POST /api/index first")
    photos, total = adapters.list_photos(offset, limit)
    return {"total": total, "offset": offset, "limit": limit, "photos": [p.model_dump() for p in photos]}


@router.get("/photos/{photo_id}")
def get_photo_record(photo_id: str) -> PhotoRecord:
    record = adapters.get_photo(photo_id)
    if record is None:
        raise HTTPException(404, "photo not found")
    return record


@router.get("/photos/{photo_id}/image")
@router.get("/photo/{photo_id}")
def get_photo(photo_id: str, thumbnail: bool = False, w: int | None = None):
    """The bytes. `?w=` is a thumbnail hint from the grid; anything <= 512 gets
    the cached 320px thumbnail rather than a 12 MP original per tile."""
    path = adapters.get_photo_path(photo_id)
    if path is None or not path.exists():
        raise HTTPException(404, "photo not found")
    if not thumbnail and not (w and w <= 512):
        return FileResponse(path)
    return FileResponse(_ensure_thumbnail(photo_id, path))


def _ensure_thumbnail(photo_id: str, original: Path) -> Path:
    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    thumb_path = THUMBNAIL_DIR / f"{photo_id}.jpg"
    if thumb_path.exists() and thumb_path.stat().st_mtime >= original.stat().st_mtime:
        return thumb_path
    with Image.open(original) as im:
        im = im.convert("RGB")
        im.thumbnail(THUMBNAIL_SIZE)
        im.save(thumb_path, "JPEG", quality=85)
    return thumb_path


# ---------------------------------------------------------------------------
# Query / feedback
# ---------------------------------------------------------------------------


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if not adapters.index_exists():
        raise HTTPException(409, "no index yet — POST /api/index first")
    if adapters.get_photo(req.source_photo_id) is None:
        raise HTTPException(404, "source_photo_id not in index")
    return adapters.run_query(req.source_photo_id, req.box, req.top_k, req.object_id)


@router.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    exemplars = adapters.submit_feedback(req.query_id, req.photo_id, req.confirmed, req.box)
    return {"ok": True, "exemplars": exemplars}
