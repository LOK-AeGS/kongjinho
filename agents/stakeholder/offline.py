"""합성 fixture를 이용한 네트워크 없는 재현 테스트용 백엔드."""
import json
from copy import deepcopy
from pathlib import Path

from .models import Extraction


class FixtureBackend:
    def __init__(self, path):
        self.data = json.loads(Path(path).read_text(encoding="utf-8"))

    def search(self, request, queries, technical_findings):
        batch = deepcopy(self.data["batch"])
        ids = {q["id"] for q in queries}
        batch["search_logs"] = [q for q in batch["search_logs"] if q["id"] in ids]
        allowed = {u for q in batch["search_logs"] for u in q["urls"]}
        batch["pages"] = {u: p for u, p in batch["pages"].items() if u in allowed}
        return batch

    def extract(self, request, batches, feedback):
        urls = {u for b in batches for u in b["pages"]}
        result = Extraction.model_validate(self.data["extraction"])
        result.observations = [o for o in result.observations if o.source_url in urls]
        return result
