"""The shared contract between indexing, retrieval, api, and the frontend.

Every other module imports these models instead of inventing its own dicts.
If a field is missing, add it here first — don't shadow it downstream.

COORDINATE CONVENTION (read this before touching any box/bbox field):
  All boxes are NORMALIZED xyxy: (x1, y1, x2, y2), each in [0, 1],
  relative to the image's (width, height), with the ORIGIN AT TOP-LEFT
  and Y increasing DOWNWARD (standard image convention, not math/plot
  convention). x1 < x2 and y1 < y2. To get pixel coords: multiply x by
  image width and y by image height.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class BoxXYXY(BaseModel):
    """Normalized box, origin top-left, xyxy order. See module docstring."""

    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)


class GpsCoord(BaseModel):
    lat: float
    lon: float


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


class IndexRequest(BaseModel):
    """Point the indexer at a folder. Video files inside it are frame-sampled."""

    folder_path: str
    video_frame_interval_sec: float = Field(
        default=2.0, description="Sample one frame every N seconds from videos."
    )


class IndexStage(str, Enum):
    scanning = "scanning"
    extracting_frames = "extracting_frames"
    embedding = "embedding"
    building_index = "building_index"
    done = "done"
    error = "error"


class IndexProgress(BaseModel):
    """Polled (or streamed) while a folder is being indexed."""

    job_id: str
    stage: IndexStage
    total_files: int = 0
    processed_files: int = 0
    total_frames: int = 0
    processed_frames: int = 0
    message: str = ""
    error: str | None = None


class PhotoRecord(BaseModel):
    """One indexed image, or one sampled video frame."""

    id: str
    path: str
    source_video_path: str | None = Field(
        default=None, description="Set when this record is a sampled video frame."
    )
    frame_time_sec: float | None = Field(
        default=None, description="Timestamp within the source video, if applicable."
    )
    taken_at: datetime | None = None
    gps: GpsCoord | None = None
    width: int
    height: int


# ---------------------------------------------------------------------------
# Query / retrieval
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """A user drew a box around an object in one photo; find that object elsewhere."""

    source_photo_id: str
    box: BoxXYXY
    top_k: int = Field(default=50, description="Recall-stage candidates before verification.")
    object_id: str | None = Field(
        default=None,
        description=(
            "A previous QueryResponse.query_id. Pass it back to search with everything the "
            "user has confirmed/rejected for that object so far — this is what makes the "
            "confirm/reject loop actually improve results instead of just being recorded."
        ),
    )


class Match(BaseModel):
    """One candidate photo that survived geometric verification, or is pending it."""

    photo_id: str
    score: float = Field(description="Stage-1 recall similarity, 0-1.")
    inlier_count: int = Field(
        default=0, description="Stage-2 RANSAC inlier count; the real confidence signal."
    )
    verified: bool = Field(
        default=False, description="Passed the geometric-verification inlier threshold."
    )
    bbox: BoxXYXY = Field(description="Matched region in this candidate photo, normalized xyxy.")
    taken_at: datetime | None = None
    gps: GpsCoord | None = None


class QueryResponse(BaseModel):
    query_id: str
    matches: list[Match] = Field(
        default_factory=list, description="Verified matches, sorted chronologically ascending by taken_at."
    )
    last_seen: Match | None = Field(
        default=None, description="Most recent verified match — the answer to 'where did I leave it'."
    )


# ---------------------------------------------------------------------------
# Feedback (confirm/reject loop)
# ---------------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    query_id: str
    photo_id: str
    confirmed: bool
    box: BoxXYXY | None = Field(
        default=None,
        description=(
            "The Match.bbox being confirmed. Supplying it means the backend stores exactly the "
            "region the user endorsed; without it the backend must re-verify to locate the object."
        ),
    )
