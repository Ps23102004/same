"""README's headline table, as a runnable command.

eval.py measures the pipeline in-process. This measures it THROUGH THE RUNNING
HTTP API, over one index holding the easy and hard suites at once -- so every
query faces same-class traps from both suites, not just its own.

    # terminal 1
    SAME_INDEX_DIR=/tmp/same_http/index \
      uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
    # terminal 2
    uv run python -m backend.app.retrieval.http_eval --build   # first time
    uv run python -m backend.app.retrieval.http_eval

`--build` renders the two suites and indexes them into SAME_INDEX_DIR; without
it the script assumes the index is already there and only queries.

Protocol: leave-one-out over the 8 instance-A photos of each suite, using the
ground-truth box. Every other A photo of that suite is a true positive (7 per
query, 56 per suite). Anything else the API returns is a false positive, split
into same-class (that suite's B/C instances) and everything else.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

API = os.environ.get("SAME_API", "http://127.0.0.1:8000")
WORK = Path(os.environ.get("SAME_HTTP_EVAL_DIR", "/tmp/same_http"))
SUITE_NAMES = ("easy", "hard")


def _post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        API + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def _get(path: str) -> dict:
    with urllib.request.urlopen(API + path, timeout=120) as r:
        return json.loads(r.read())


def build() -> dict:
    """Render both suites and index them into ONE index. Returns the manifests.

    The easy suite is built without its twin instance here: the twins are a
    separate question (eval.py's report_twins) and counting them as either
    true or false positives would misstate this table.
    """
    import shutil

    from backend.app.indexing.indexer import index_folder
    from backend.app.retrieval.evalset import SUITES, build as build_suite

    photos = WORK / "photos"
    index_dir = Path(os.environ.get("SAME_INDEX_DIR", WORK / "index"))
    shutil.rmtree(photos, ignore_errors=True)
    shutil.rmtree(index_dir, ignore_errors=True)
    mans = {
        "easy": build_suite(photos / "easy", "easy", cfg=replace(SUITES["easy"], twins=False)),
        "hard": build_suite(photos / "hard", "hard"),
    }
    (WORK / "manifests.json").write_text(json.dumps(mans))
    stats = index_folder(photos, index_dir)
    print(f"indexed {stats.photos_indexed} photos in {stats.seconds:.1f}s "
          f"({stats.images_per_sec:.2f} img/s) -> {index_dir}")
    return mans


def _instance(path: str) -> str | None:
    name = Path(path).name
    return None if name.startswith("unrelated") else name[0]


def main() -> None:
    if "--build" in sys.argv:
        build()
    mans = json.loads((WORK / "manifests.json").read_text())

    photos = _get("/api/photos?offset=0&limit=10000")["photos"]
    by_path = {str(Path(p["path"]).resolve()): p["id"] for p in photos}
    id2path = {v: k for k, v in by_path.items()}
    print(f"index holds {len(photos)} photos  ({API})")

    latencies: list[float] = []
    for suite in SUITE_NAMES:
        man = mans[suite]
        qnames = sorted(n for n in man if man[n]["instance"] == "A")
        # Same-class traps this suite's queries are exposed to, per query.
        n_traps = sum(
            1 for p in photos
            if Path(p["path"]).parent.name == suite and _instance(p["path"]) in ("B", "C")
        )
        n_other = len(photos) - n_traps - len(qnames)

        tp = tp_confirmed = fp_same_class = fp_other = 0
        for qname in qnames:
            qid = by_path[str((WORK / "photos" / suite / qname).resolve())]
            x1, y1, x2, y2 = man[qname]["box"]
            body = {"source_photo_id": qid,
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, "top_k": 200}
            t0 = time.time()
            r = _post("/api/query", body)
            latencies.append(time.time() - t0)

            truth = {str((WORK / "photos" / suite / n).resolve())
                     for n in qnames if n != qname}
            got = {id2path[m["photo_id"]] for m in r["matches"]}
            tp += len(got & truth)
            for wrong in got - truth:
                same_suite = Path(wrong).parent.name == suite
                if same_suite and _instance(wrong) in ("B", "C"):
                    fp_same_class += 1
                else:
                    fp_other += 1

            # ...then confirm every true positive it found and ask again.
            oid = r["query_id"]
            for m in r["matches"]:
                if id2path[m["photo_id"]] in truth:
                    _post("/api/feedback", {"query_id": oid, "photo_id": m["photo_id"],
                                            "confirmed": True, "box": m["bbox"]})
            r2 = _post("/api/query", dict(body, object_id=oid))
            tp_confirmed += len({id2path[m["photo_id"]] for m in r2["matches"]} & truth)

        n_pos = len(qnames) * (len(qnames) - 1)
        print(f"\n{suite}: {len(qnames)} queries, leave-one-out")
        print(f"  recall                    {tp}/{n_pos} = {tp / n_pos:.3f}")
        print(f"  recall after confirming   {tp_confirmed}/{n_pos} = {tp_confirmed / n_pos:.3f}")
        print(f"  same-class accepted       {fp_same_class} / {len(qnames) * n_traps}")
        print(f"  other accepted            {fp_other} / {len(qnames) * n_other}")

    print(f"\nquery latency: median {statistics.median(latencies):.2f} s, "
          f"min {min(latencies):.2f} s, max {max(latencies):.2f} s, n={len(latencies)}")


if __name__ == "__main__":
    main()
