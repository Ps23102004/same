"""Fast assert-based check that the two-stage pipeline is actually working.

    uv run python -m backend.app.retrieval.selfcheck      # ~30 s incl. model load

Not a replacement for eval.py (which measures precision/recall properly); this
is the smallest thing that fails loudly if retrieval breaks. It builds real
images, runs the real backbone, and asserts the one property the product lives
or dies by: same instance in, same-class lookalike out.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from backend.app.indexing import features as F
from backend.app.indexing.indexer import index_folder
from backend.app.retrieval.evalset import SUITES, build
from backend.app.retrieval.search import Searcher


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="same_selfcheck_"))
    try:
        cfg = replace(SUITES["easy"], n_per_instance=3, n_unrelated=2, twins=False)
        manifest = build(tmp / "photos", "easy", cfg=cfg)
        embedder = F.Embedder()
        print(f"device: {embedder.device}")
        index_folder(tmp / "photos", tmp / "index", embedder=embedder)
        s = Searcher(tmp / "index", embedder=embedder)
        name = {p.id: Path(p.path).name for p in s.store.all_records()}
        id_of = {v: k for k, v in name.items()}

        resp, cands, warning = s.query_detailed(id_of["A_00.jpg"], manifest["A_00.jpg"]["box"])
        got = {name[m.photo_id] for m in resp.matches}
        print(f"  matches: {sorted(got)}")
        print(f"  inliers: {[(name[m.photo_id], m.inlier_count) for m in resp.matches]}")

        assert got >= {"A_01.jpg", "A_02.jpg"}, f"missed the same instance: {got}"
        assert not any(n[0] in "BC" for n in got), f"returned a same-class lookalike: {got}"
        assert not any(n.startswith("unrelated") for n in got), f"returned junk: {got}"
        assert resp.matches == sorted(resp.matches, key=lambda m: m.taken_at), "not chronological"
        assert resp.last_seen and resp.last_seen.verified
        assert resp.last_seen.taken_at == max(m.taken_at for m in resp.matches)
        assert all(0 <= v <= 1 for m in resp.matches
                   for v in (m.bbox.x1, m.bbox.y1, m.bbox.x2, m.bbox.y2))
        assert warning is None, f"unexpected duplicate warning: {warning}"

        # feedback round-trip
        n = s.feedback(resp.query_id, resp.matches[0].photo_id, True, resp.matches[0].bbox)
        assert n >= 2, n
        s.feedback(resp.query_id, resp.matches[0].photo_id, False)
        again = s.query(id_of["A_00.jpg"], manifest["A_00.jpg"]["box"], object_id=resp.query_id)
        assert resp.matches[0].photo_id not in {m.photo_id for m in again.matches}, "reject ignored"

        print("selfcheck OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
