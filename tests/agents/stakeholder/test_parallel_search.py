"""웹 검색 호출 병렬화 회귀 테스트. 네트워크 없이 가짜 client로 실행한다."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agents.stakeholder.backend import OpenAIBackend  # noqa: E402


class FakeResponse:
    status = "completed"

    def __init__(self, query_id):
        self.id = f"resp-{query_id}"
        self.output_text = f"notes {query_id}"
        self._url = f"https://example.com/{query_id}"

    def model_dump(self):
        return {"output": [
            {"type": "web_search_call", "status": "completed",
             "action": {"type": "search", "sources": [{"url": self._url}]}},
        ]}


class SlowClient:
    """호출 하나에 0.3초가 걸리는 가짜 Responses API."""

    def __init__(self):
        self.responses = self

    def create(self, **kwargs):
        import json
        time.sleep(0.3)
        return FakeResponse(json.loads(kwargs["input"])["query"]["id"])


class OkFetcher:
    def fetch(self, url):
        return {"url": url, "status": "ok", "blocks": [{"locator": "b1", "text": "x" * 120}]}


def test_search_calls_run_in_parallel_and_logs_keep_query_order(tmp_path):
    backend = OpenAIBackend(client=SlowClient(), fetcher=OkFetcher(), cache_dir=tmp_path)
    backend._pace = lambda: None  # 시작 간격 1초를 빼고 병렬 효과만 본다
    queries = [{"id": f"q{index}", "query": f"query {index}"} for index in range(6)]
    started = time.monotonic()
    batch = backend.search({"as_of_date": "2026-09-22"}, queries, {})
    elapsed = time.monotonic() - started
    assert elapsed < 6 * 0.3 * 0.6  # 순차라면 1.8초
    assert [log["id"] for log in batch["search_logs"]] == [query["id"] for query in queries]
    assert all(log["status"] == "ok" for log in batch["search_logs"])
    assert set(batch["pages"]) == {f"https://example.com/q{index}" for index in range(6)}
