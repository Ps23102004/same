"""Two-stage instance retrieval: DINOv2 recall + RANSAC geometric verification.

Deliberately does NOT re-export `verify.verify` at package level -- that name
would shadow the `verify` submodule and silently break `from ... import verify`.
"""

from backend.app.retrieval.search import DEFAULT_TOP_K, INLIER_THRESHOLD, Candidate, Searcher

__all__ = ["Searcher", "Candidate", "INLIER_THRESHOLD", "DEFAULT_TOP_K"]
