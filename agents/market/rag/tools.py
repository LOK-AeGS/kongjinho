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


# 출처 등급 필터가 등급표 밖 도메인을 전부 기각하므로, 질의당 3건만 받으면 통과하는
# 결과가 거의 남지 않는다. 실측(질의 8건 기준): 3건씩 받으면 24건 중 5~6건만 등급을
# 통과하고 최종 근거가 0~3건으로 실행마다 출렁였다. 10건씩 받으면 80건 중 22건이
# 통과하고 근거 6건·주장 7건으로 안정됐다. 질의 언어(한국어/영어)는 영향이 없었다.
def tavily_web_search(query: str, max_results: int = 10) -> list[SearchResult]:
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
