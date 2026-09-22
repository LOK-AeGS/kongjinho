"""웹 검색 기본 구현. 전역 상태 없이 함수 형태로 두고 make_node()에서 deps로 주입한다."""

from __future__ import annotations

import hashlib
from urllib.parse import urlparse

from agents.market.state import SearchResult


def stub_web_search(query: str) -> list[SearchResult]:
    """API 키 없이 오프라인 테스트를 돌리기 위한 가짜 검색."""
    h = hashlib.md5(query.encode()).hexdigest()[:8]
    return [{
        "title": f"[STUB] {query}", "url": f"https://example.com/stub/{h}",
        "content": f"[STUB] '{query}'에 대한 가짜 검색 결과이며 실제 근거가 아니다.",
        "organization": "example.com", "published_date": None, "source_type": "news",
    }]


def tavily_web_search(query: str, max_results: int = 3) -> list[SearchResult]:
    from langchain_tavily import TavilySearch

    tool = TavilySearch(max_results=max_results)
    raw = tool.invoke({"query": query})
    return [
        {
            "title": r.get("title", ""), "url": r["url"], "content": r.get("content", ""),
            "organization": urlparse(r["url"]).netloc, "published_date": None, "source_type": "other",
        }
        for r in raw.get("results", [])
    ]
