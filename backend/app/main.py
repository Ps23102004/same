"""Same backend entrypoint.

Bound to 127.0.0.1 only (never 0.0.0.0) — this is a local-first privacy
product; nothing about a photo library should ever be reachable off the
machine, and binding to all interfaces would contradict the entire premise.

Run: uv run uvicorn backend.app.main:app --port 8000
(host is pinned below via `if __name__ == "__main__"`; if you invoke
uvicorn directly, pass --host 127.0.0.1 yourself.)
"""

from __future__ import annotations

from fastapi import FastAPI

from backend.app.api.routes import router

app = FastAPI(title="Same")
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
