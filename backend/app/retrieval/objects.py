"""What a confirm/reject tap actually changes.

An object's model is a set of EXEMPLAR CROPS, not a single embedding. Confirming
a result appends that photo's verified region as another exemplar; the next
query embeds every exemplar and takes the best score per photo, so an object
confirmed once from the side becomes findable from the side. Rejecting
blacklists that photo for that object, so a stubborn false positive stops
coming back.

This is measured, not asserted -- eval.py reports recall before and after a
single confirm on the hardest (lowest-scoring) true positive.

Lives at <index_dir>/objects.json. Deliberately separate from the indexer's
files so re-indexing a folder never wipes what the user taught it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ObjectMemory:
    object_id: str
    exemplars: list[dict] = field(default_factory=list)  # {"photo_id": str, "box": [x1,y1,x2,y2]}
    rejected: list[str] = field(default_factory=list)


class ObjectStore:
    def __init__(self, index_dir: str | Path):
        self.path = Path(index_dir) / "objects.json"
        raw = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.objects: dict[str, ObjectMemory] = {
            k: ObjectMemory(**v) for k, v in raw.items()
        }

    def get(self, object_id: str) -> ObjectMemory:
        return self.objects.setdefault(object_id, ObjectMemory(object_id=object_id))

    def add_exemplar(self, object_id: str, photo_id: str, box) -> ObjectMemory:
        mem = self.get(object_id)
        mem.rejected = [p for p in mem.rejected if p != photo_id]
        if not any(e["photo_id"] == photo_id for e in mem.exemplars):
            mem.exemplars.append({"photo_id": photo_id, "box": [float(v) for v in box]})
        return mem

    def reject(self, object_id: str, photo_id: str) -> ObjectMemory:
        mem = self.get(object_id)
        mem.exemplars = [e for e in mem.exemplars if e["photo_id"] != photo_id]
        if photo_id not in mem.rejected:
            mem.rejected.append(photo_id)
        return mem

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: asdict(v) for k, v in self.objects.items()}, indent=1))
        tmp.replace(self.path)
