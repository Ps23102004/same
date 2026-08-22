"""Prove the privacy claim instead of asserting it.

Runs a REAL index pass and a REAL query with a CPython audit hook watching
`socket.connect`. Every connection attempt is recorded; anything that is not a
loopback address is a violation and fails the script.

    uv run python scripts/verify_offline.py [folder]

The model must already be in the local HuggingFace cache (see README) -- which
is the point: with SAME_ALLOW_MODEL_DOWNLOAD unset, Same loads it with
local_files_only and never reaches for the network.
"""

from __future__ import annotations

import ipaddress
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONNECTIONS: list[str] = []


def _host_of(address) -> str | None:
    if isinstance(address, tuple) and address and isinstance(address[0], str):
        return address[0]
    return None  # AF_UNIX and friends: a filesystem path, not a network hop


def _audit(event: str, args) -> None:
    if event != "socket.connect":
        return
    host = _host_of(args[1] if len(args) > 1 else None)
    if host:
        CONNECTIONS.append(host)


sys.addaudithook(_audit)

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "sample_photos"
index_dir = ROOT / ".index-offline-check"

from backend.app.indexing import IndexStore, index_folder  # noqa: E402
from backend.app.retrieval.search import Searcher  # noqa: E402

print(f"indexing {folder} -> {index_dir}")
stats = index_folder(folder, index_dir=index_dir)
print(f"  {stats.total_photos} photos, {stats.seconds:.1f}s")

store = IndexStore(index_dir)
first = sorted(store.all_records(), key=lambda r: r.path)[0]
print(f"querying a crop of {Path(first.path).name}")
resp = Searcher(index_dir).query(first.id, (0.3, 0.3, 0.7, 0.7))
print(f"  {len(resp.matches)} matches")

offsite = []
for host in CONNECTIONS:
    try:
        if not ipaddress.ip_address(host).is_loopback:
            offsite.append(host)
    except ValueError:
        offsite.append(host)  # a hostname means DNS, which means off-machine

print(f"\nsocket.connect calls during index + query: {len(CONNECTIONS)}")
for h in sorted(set(CONNECTIONS)):
    print(f"  {h}")
if offsite:
    print(f"\nFAIL: {len(offsite)} non-loopback connection(s): {sorted(set(offsite))}")
    raise SystemExit(1)
print("\nOK: nothing left this machine.")
