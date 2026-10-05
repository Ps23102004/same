"""evalset.build must be byte-for-byte reproducible for a given seed.

The README's headline numbers depend on this: an earlier version seeded scene
layout with hash(str), which CPython salts per process, so `hard` recall swung
between runs.
"""
import hashlib
from dataclasses import replace
from pathlib import Path

from backend.app.retrieval.evalset import SUITES, build

CFG = replace(SUITES["easy"], n_per_instance=2, n_unrelated=1, twins=False)


def digest(folder: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob("*.jpg"))}


def test_same_seed_same_bytes(tmp_path):
    build(tmp_path / "a", "easy", seed=7, cfg=CFG)
    build(tmp_path / "b", "easy", seed=7, cfg=CFG)
    assert digest(tmp_path / "a") == digest(tmp_path / "b")
    assert len(digest(tmp_path / "a")) > 0


def test_seed_reaches_the_scenes(tmp_path):
    build(tmp_path / "a", "easy", seed=7, cfg=CFG)
    build(tmp_path / "b", "easy", seed=8, cfg=CFG)
    assert digest(tmp_path / "a") != digest(tmp_path / "b")
