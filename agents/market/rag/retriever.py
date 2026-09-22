"""팀 공유(또는 개발용) RAG 검색기와 시장 에이전트를 잇는 어댑터.

검색기 계약: retriever(query, tech_id, k) -> list[Chunk]. 이 계약만 맞으면 어떤 구현이든 꽂을 수 있다.
"""

from __future__ import annotations

import re
from typing import Callable, TypedDict

from agents.market.state import SearchResult


class Chunk(TypedDict):
    text: str
    document_id: str
    title: str
    page: int
    section: str | None
    tech_id: str  # "sw" / "hw"
    source_url: str
    published_date: str | None


Retriever = Callable[[str, str | None, int], list[Chunk]]


def chunk_to_result(c: Chunk) -> SearchResult:
    return {
        "title": f"{c['title']} (p.{c['page']})", "url": c["source_url"], "content": c["text"],
        "organization": "논문 원문", "published_date": c["published_date"], "source_type": "paper",
        "document_id": c["document_id"], "page": c["page"], "section": c["section"],
    }


def make_rag_fn(retriever: Retriever, k: int = 3) -> Callable[[str, str | None], list[SearchResult]]:
    def rag(query: str, tech_id: str | None = None) -> list[SearchResult]:
        return [chunk_to_result(c) for c in retriever(query, tech_id, k)]

    return rag


FAKE_CHUNKS: list[Chunk] = [
    {
        "text": "[FAKE] The MLA prototype is described as being in production deployment for a hosted API service.",
        "document_id": "deepseek-v2", "title": "DeepSeek-V2", "page": 3, "section": "Introduction",
        "tech_id": "sw", "source_url": "https://arxiv.org/abs/2405.04434", "published_date": "2024-06",
    },
    {
        "text": "[FAKE] ITME is evaluated on an FPGA prototype; cache-miss bandwidth stays below production CMM levels.",
        "document_id": "itme", "title": "ITME", "page": 10, "section": "Limitations",
        "tech_id": "hw", "source_url": "https://arxiv.org/abs/2606.12556", "published_date": "2026-06",
    },
]


def fake_retriever(query: str, tech_id: str | None, k: int) -> list[Chunk]:
    """오프라인 테스트용. 질의 단어와 겹치는 정도로 순위를 매긴다."""
    words = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = []
    for c in FAKE_CHUNKS:
        if tech_id and c["tech_id"] != tech_id:
            continue
        scored.append((len(words & set(re.findall(r"[a-z0-9]+", c["text"].lower()))), c))
    return [c for _, c in sorted(scored, key=lambda x: -x[0])[:k]]
