"""Local indexing: folder -> DINOv2 region descriptors on disk.

Public API (this is what retrieval/ and api/ should import):

    from backend.app.indexing import (
        IndexStore, IndexStats, RegionHit, PhotoEntry,
        index_folder, open_image, Embedder, region_boxes,
    )

    stats = index_folder(folder, index_dir=None, video_frame_interval_sec=2.0,
                         progress=cb, job_id="...", embedder=None) -> IndexStats
    store = IndexStore(index_dir_or_None)
    store.search(query_vec_or_vecs, top_k_photos=50) -> list[RegionHit]
    store.get(photo_id) -> PhotoEntry | None       (.record -> schemas.PhotoRecord)
    store.all_records() -> list[PhotoRecord]
    store.sift_for(photo_id) -> (xy (N,2) float32 px, desc (N,128) float32)

    emb = Embedder()                                # loads dinov2-base once
    emb.embed_crops(pil_image, boxes_norm_xyxy) -> (K, 768) float32, unit norm
    features.sift_features(pil_image) -> (xy, desc) for an unindexed image/crop
"""

from .features import DIM, GRID, IMAGE_SIZE, MODEL_ID, N_REGIONS, Embedder, region_boxes, sift_features
from .indexer import IndexStats, index_folder, read_exif, sample_video
from .store import IndexStore, PhotoEntry, RegionHit, default_index_dir, open_image

__all__ = [
    "DIM", "GRID", "IMAGE_SIZE", "MODEL_ID", "N_REGIONS",
    "Embedder", "region_boxes", "sift_features",
    "IndexStats", "index_folder", "read_exif", "sample_video",
    "IndexStore", "PhotoEntry", "RegionHit", "default_index_dir", "open_image",
]
