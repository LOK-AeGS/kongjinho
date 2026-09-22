"""긴 문서 청킹·색인 (Pool B — 에이전트 전용 인덱스 B).

fetch.py가 가져온 본문이 SHORT_DOCUMENT_CHARS를 넘으면 이 모듈로 청킹한 뒤 BM25로
질의와 맞는 조각만 근거 후보로 올린다. 짧은 문서는 이 단계를 건너뛴다(index.py 미사용).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agents.market.rag.fetch import FetchedDocument

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200


@dataclass
class Chunk:
    text: str
    locator: str  # fetch.DocumentPart.locator 상속 (예: "p.7", "chars:0-1200")
    url: str
    title: str


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9가-힣]+", text.lower())


def chunk_document(doc: FetchedDocument) -> list[Chunk]:
    """part 경계를 넘어서까지 CHUNK_CHARS 단위로 겹쳐 자른다. locator는 원래 part 것을 쓴다."""
    chunks: list[Chunk] = []
    for part in doc.parts:
        text = part.text
        if len(text) <= CHUNK_CHARS:
            chunks.append(Chunk(text=text, locator=part.locator, url=doc.url, title=doc.title))
            continue
        step = CHUNK_CHARS - CHUNK_OVERLAP
        for i in range(0, len(text), step):
            piece = text[i : i + CHUNK_CHARS]
            if len(piece) < 100:  # 겹침 때문에 생기는 꼬리 조각은 버린다.
                continue
            chunks.append(Chunk(text=piece, locator=part.locator, url=doc.url, title=doc.title))
    return chunks


def search_chunks(chunks: list[Chunk], query: str, k: int = 3) -> list[Chunk]:
    """BM25로 질의와 가장 관련 있는 조각 k개를 고른다. 조각이 적으면 BM25 없이 그대로 반환."""
    if not chunks:
        return []
    if len(chunks) <= k:
        return chunks
    from rank_bm25 import BM25Okapi

    corpus = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(scores, chunks), key=lambda x: -x[0])
    return [c for _, c in ranked[:k]]
