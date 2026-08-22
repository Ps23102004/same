"""Honest measurement of the two-stage pipeline. Run it; believe the numbers.

    uv run python -m backend.app.retrieval.eval            # all three suites
    uv run python -m backend.app.retrieval.eval hard       # one suite

What it does, per suite:

  1. Generates a labelled set (evalset.py) and indexes it with the real indexer.
  2. Uses each photo of instance A in turn as the query, with the ground-truth
     box, leave-one-out. 8 queries per suite.
  3. Labels every returned candidate: `same` (instance A), `same_class`
     (instance B/C -- the thing that must be rejected), `unrelated`.
  4. Reports the STAGE-1 ONLY baseline. This is the "what if we just used a
     global embedding" control, and it is the reason stage 2 exists.
  5. Sweeps the inlier threshold and prints the precision/recall curve, then
     picks the operating point off that curve.
  6. Checks the confirm-feedback loop actually improves recall.
  7. Checks the timeline is chronological and `last_seen` is the most recent
     VERIFIED match.

Nothing here is mocked. Delete the index dir and it rebuilds from pixels.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from backend.app.indexing import features as F
from backend.app.indexing.indexer import index_folder
from backend.app.retrieval.evalset import SUITES, build
from backend.app.retrieval.search import MIN_INLIER_FRACTION, MIN_INLIERS, STRONG_INLIERS, Searcher

TOP_K = 200
SWEEP = [4, 5, 6, 8, 10, 12, 15, 20, 30, 60, 100]
FRACTIONS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]


def label_of(manifest: dict, name: str) -> str:
    inst = manifest[name]["instance"]
    if inst == "A":
        return "same"
    if inst == "T":
        return "twin"          # pixel-identical second copy; see report_twins()
    return "same_class" if inst else "unrelated"


def run_suite(suite: str, root: Path, embedder: F.Embedder) -> dict:
    photos = root / "photos"
    index_dir = root / "index"
    shutil.rmtree(index_dir, ignore_errors=True)
    manifest = build(photos, suite)
    stats = index_folder(photos, index_dir, embedder=embedder)
    print(f"  indexed {stats.photos_indexed} photos in {stats.seconds:.1f}s "
          f"({stats.images_per_sec:.1f} img/s)")

    s = Searcher(index_dir, embedder=embedder)
    name_of = {p.id: Path(p.path).name for p in s.store.all_records()}
    id_of = {v: k for k, v in name_of.items()}
    queries = sorted(n for n in manifest if manifest[n]["instance"] == "A")

    rows = []          # (label, inliers, recall_score)
    stage1_rows = []   # (label, rank) from stage-1 alone
    n_positives = 0

    for qname in queries:
        qid = id_of[qname]
        cands = s.search(qid, manifest[qname]["box"], top_k=TOP_K)
        # leave-one-out: every OTHER A photo is a true positive for this query
        n_positives += sum(1 for n in queries if n != qname)
        seen = set()
        for c in cands:
            n = name_of[c.photo.id]
            seen.add(n)
            rows.append((label_of(manifest, n), c.inliers, c.raw_matches, c.recall_score))
        for rank, hit in enumerate(sorted(cands, key=lambda c: -c.recall_score)):
            stage1_rows.append((label_of(manifest, name_of[hit.photo.id]), rank))
        missed = [n for n in queries if n != qname and n not in seen]
        if missed:
            print(f"  STAGE-1 MISS for {qname}: {missed}")

    return {
        "suite": suite,
        "rows": rows,
        "stage1_rows": stage1_rows,
        "n_positives": n_positives,
        "n_queries": len(queries),
        "searcher": s,
        "manifest": manifest,
        "id_of": id_of,
        "name_of": name_of,
        "queries": queries,
    }


def stage1_baseline(res: dict) -> None:
    """Precision@k of stage 1 alone -- i.e. what a global-embedding product ships."""
    print("\n  STAGE 1 ONLY (no geometric verification) -- the control:")
    print("    k    precision   what the extra results are")
    for k in (5, 7, 10, 20):
        top = [lbl for lbl, rank in res["stage1_rows"] if rank < k]
        c = Counter(top)
        prec = c["same"] / max(len(top), 1)
        print(f"    {k:<4} {prec:6.3f}      same_class={c['same_class']:<4} unrelated={c['unrelated']}")


def sweep(res: dict) -> tuple[int, float, list]:
    """Precision/recall over BOTH gates. Twins are excluded from the counts --
    they are neither a true positive nor a false one, they are the open problem
    (report_twins covers them)."""
    rows = [r for r in res["rows"] if r[0] != "twin"]
    n_pos = res["n_positives"]
    print("\n  STAGE 1 + 2, gate sweep (inliers >= thr AND inliers/matches >= frac):")
    print("    thr  frac   TP    FP(same_class)  FP(unrelated)  precision  recall   F1")
    curve = []
    for frac in FRACTIONS:
        for thr in SWEEP:
            kept = [r for r in rows if r[1] >= thr and r[1] / max(r[2], 1) >= frac]
            c = Counter(l for l, *_ in kept)
            tp, fp = c["same"], c["same_class"] + c["unrelated"]
            prec = tp / max(tp + fp, 1)
            rec = tp / max(n_pos, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            curve.append((thr, frac, tp, c["same_class"], c["unrelated"], prec, rec, f1))
    # print the frac=0 row (pure inlier count, the naive design) and the chosen
    # frac, so the reader can see what the second gate buys.
    for frac in (0.0, MIN_INLIER_FRACTION):
        for row in [c for c in curve if c[1] == frac]:
            print(f"    {row[0]:<4} {row[1]:<6} {row[2]:<5} {row[3]:<15} {row[4]:<14} "
                  f"{row[5]:9.3f}  {row[6]:6.3f}  {row[7]:5.3f}")
        print()
    clean = [c for c in curve if c[5] >= 1.0 and c[2] > 0]
    best = max(clean, key=lambda c: (c[6], -c[1])) if clean else max(curve, key=lambda c: c[7])
    print(f"  -> best zero-false-positive gate on THIS suite: inliers >= {best[0]}, "
          f"frac >= {best[1]}  (recall {best[6]:.3f})")
    print(f"  -> SHIPPED gate (fixed across all suites): inliers >= {MIN_INLIERS}, "
          f"frac >= {MIN_INLIER_FRACTION}, or inliers >= {STRONG_INLIERS}")
    return MIN_INLIERS, MIN_INLIER_FRACTION, curve


def per_class_table(res: dict) -> None:
    s: Searcher = res["searcher"]
    rows = res["rows"]
    print(f"\n  PER-CLASS at the shipped gate ({res['n_queries']} queries, leave-one-out):")
    print("    class                          candidates  accepted   accept-rate")
    for lbl, human in (("same", "(a) same instance"),
                       ("same_class", "(b) same class, diff instance"),
                       ("unrelated", "(c) unrelated"),
                       ("twin", "(d) identical twin object")):
        sub = [r for r in rows if r[0] == lbl]
        if not sub:
            continue
        acc = sum(1 for r in sub if s.accept(r[1], r[2]))
        print(f"    {human:<31}{len(sub):<12}{acc:<10}{acc / len(sub):.3f}")
    for lbl, human in (("same", "same instance"), ("same_class", "same class (traps)")):
        sub = sorted(r[1] for r in rows if r[0] == lbl)
        if sub:
            print(f"    inliers, {human:<20} min={sub[0]} median={sub[len(sub) // 2]} max={sub[-1]}")


def report_twins(res: dict) -> None:
    """The two-identical-mugs case, measured rather than hand-waved."""
    s: Searcher = res["searcher"]
    twins = [r for r in res["rows"] if r[0] == "twin"]
    if not twins:
        return
    acc = sum(1 for r in twins if s.accept(r[1], r[2]))
    print(f"\n  IDENTICAL TWIN OBJECT: {acc}/{len(twins)} accepted as the query object "
          f"({acc / len(twins):.0%}).")
    print("    Expected, and not a bug to be fixed at this layer: the twin is the same")
    print("    manufactured item, so the pixels carry no evidence that it is a different")
    print("    physical thing. See README 'Two of the same mug'.")


def timeline_check(res: dict) -> None:
    """The timeline half: chronological order, and `last_seen` = most recent
    VERIFIED match rather than most recent candidate."""
    s: Searcher = res["searcher"]
    manifest, id_of, name_of = res["manifest"], res["id_of"], res["name_of"]
    qname = res["queries"][0]
    r, cands, warning = s.query_detailed(id_of[qname], manifest[qname]["box"], top_k=TOP_K)
    print(f"\n  TIMELINE for {qname}: {len(r.matches)} verified of {len(cands)} candidates")
    print(f"    {[m.taken_at.date().isoformat() for m in r.matches]}")
    assert r.matches == sorted(r.matches, key=lambda m: m.taken_at), "not chronological"
    if r.last_seen:
        newest_candidate = max(cands, key=lambda c: c.photo.taken_at)
        print(f"    last_seen: {name_of[r.last_seen.photo_id]} @ {r.last_seen.taken_at.date()} "
              f"(inliers {r.last_seen.inlier_count})")
        print(f"    newest CANDIDATE was {name_of[newest_candidate.photo.id]} "
              f"(inliers {newest_candidate.inliers}) -- "
              f"{'same' if newest_candidate.photo.id == r.last_seen.photo_id else 'CORRECTLY IGNORED, unverified'}")
        assert r.last_seen.taken_at == max(m.taken_at for m in r.matches)
        assert r.last_seen.verified
    if warning:
        print(f"    duplicate warning: {warning}")


def feedback_experiment(res: dict) -> None:
    """Does a confirm tap actually help? Measured, over every query.

    Protocol per query: run it cold, confirm every true positive it did find
    (what a user does -- one tap each on the results that are obviously right),
    re-run, and compare recall. Objects are isolated per query so runs cannot
    contaminate each other.
    """
    s: Searcher = res["searcher"]
    manifest, id_of, name_of = res["manifest"], res["id_of"], res["name_of"]
    before = after = total = 0
    fps = 0
    for qname in res["queries"]:
        truth = {n for n in res["queries"] if n != qname}
        oid = f"exp_{qname}"
        s.objects.objects.pop(oid, None)
        r1 = s.query(id_of[qname], manifest[qname]["box"], top_k=TOP_K, object_id=oid)
        found1 = {name_of[m.photo_id] for m in r1.matches}
        for m in r1.matches:
            if name_of[m.photo_id] in truth:
                s.feedback(oid, m.photo_id, True, m.bbox)
        r2 = s.query(id_of[qname], manifest[qname]["box"], top_k=TOP_K, object_id=oid)
        found2 = {name_of[m.photo_id] for m in r2.matches}
        before += len(found1 & truth)
        after += len(found2 & truth)
        twins = {n for n in manifest if manifest[n]["instance"] == "T"}
        fps += len(found2 - truth - twins - {qname})
        total += len(truth)
        s.objects.objects.pop(oid, None)
    s.objects.save()
    print(f"\n  FEEDBACK (confirm every visible true positive, then re-query):")
    print(f"    recall {before}/{total} = {before / total:.3f}  ->  {after}/{total} = "
          f"{after / total:.3f}   false positives after: {fps}  (twins excluded)")

    # And the reject half: a blacklisted photo must not come back.
    qname = res["queries"][0]
    oid = "exp_reject"
    s.objects.objects.pop(oid, None)
    r = s.query(id_of[qname], manifest[qname]["box"], top_k=TOP_K, object_id=oid)
    if r.matches:
        victim = r.matches[0].photo_id
        s.feedback(oid, victim, False)
        r2 = s.query(id_of[qname], manifest[qname]["box"], top_k=TOP_K, object_id=oid)
        ok = victim not in {m.photo_id for m in r2.matches}
        print(f"    reject of {name_of[victim]} respected on re-query: {'yes' if ok else 'NO'}")
        assert ok
    s.objects.objects.pop(oid, None)
    s.objects.save()


def main(suites: list[str]) -> None:
    root = Path("/tmp/same_eval")
    embedder = F.Embedder()
    print(f"device: {embedder.device}")
    summary = {}
    for suite in suites:
        print(f"\n{'=' * 78}\nSUITE: {suite}  {SUITES[suite]}\n{'=' * 78}")
        res = run_suite(suite, root / suite, embedder)
        stage1_baseline(res)
        thr, frac, curve = sweep(res)
        per_class_table(res)
        report_twins(res)
        timeline_check(res)
        feedback_experiment(res)
        summary[suite] = {"gate": [thr, frac, STRONG_INLIERS], "curve": curve}
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {root / 'summary.json'}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(SUITES))
