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
        ids = {query["id"] for query in queries}
        batch["search_logs"] = [query for query in batch["search_logs"] if query["id"] in ids]
        allowed = {url for query in batch["search_logs"] for url in query["urls"]}
        batch["pages"] = {url: page for url, page in batch["pages"].items() if url in allowed}
        return batch

    def extract(self, request, batches, feedback):
        urls = {url for batch in batches for url in batch["pages"]}
        result = Extraction.model_validate(self.data["extraction"])
        result.observations = [item for item in result.observations if item.source_url in urls]
        return result
