"""Same backend entrypoint.

Bound to 127.0.0.1 only (never 0.0.0.0) — this is a local-first privacy
product; nothing about a photo library should ever be reachable off the
machine, and binding to all interfaces would contradict the entire premise.

Run: uv run uvicorn backend.app.main:app --port 8000
(host is pinned below via `if __name__ == "__main__"`; if you invoke
uvicorn directly, pass --host 127.0.0.1 yourself.)
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api import adapters
from backend.app.api.routes import router


def _warm() -> None:
    """Load dinov2 and memory-map the index before anyone asks for a search.

    Measured on this machine, 36-photo index: first query 3.13 s cold, 0.23 s
    once the model is resident. That 2.9 s used to land squarely on the user's
    first click. Doing it on a daemon thread at boot means the server answers
    /api/status immediately and is already warm by the time a photo is open.
    """
    try:
        if not (adapters.retrieval_available() and adapters.index_exists()):
            return
        searcher = adapters.searcher()
        # Touching .embedder is what actually loads dinov2 (Searcher builds it
        # lazily), and one forward pass makes the GPU compile its kernels — both
        # costs the first real query would otherwise pay in front of the user.
        from PIL import Image

        searcher.embedder.embed_images([Image.new("RGB", (448, 448))])
    except Exception:  # a broken index must not stop the server from starting
        pass


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="Same", lifespan=lifespan)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
