"""긴 문서 색인: BM25 + dense 하이브리드.

임베딩이 필요한 지점은 여기 하나다. 웹 검색이 데려온 논문·백서는 수십 페이지라
통째로 프롬프트에 넣을 수 없고, 질문에 해당하는 부분만 뽑아야 한다.
짧은 문서는 색인하지 않고 그대로 근거가 되므로 이 모듈을 거치지 않는다.

BM25를 함께 쓰는 이유는 질의 유형이 둘이기 때문이다. "메모리 압박을 어떻게 완화하나"
같은 의미 질의는 dense 가 강하고, "MLA", "CXL", "93.3%" 같은 고유명사·수치는
BM25 가 정확히 집는다. 순위 융합은 RRF 를 쓴다(점수 척도가 서로 달라 단순 합산은 위험).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field

from rag.evidence import normalize_text
from rag.fetch import DocumentPart

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
RRF_K = 60  # 순위 융합 상수. 낮을수록 상위 순위에 더 큰 가중치.


@dataclass
class Chunk:
    text: str
    url: str
    title: str
    locator: str
    chunk_index: int


@dataclass
class EmbeddingRunInfo:
    """재현성 기록용. 같은 모델이라도 device·precision 이 다르면 결과가 미세하게 달라진다."""

    model_name: str
    device: str
    normalize_embeddings: bool
    batch_size: int
    dimension: int | None = None
    torch_version: str | None = None
    platform: str = field(default_factory=lambda: f"{platform.system()} {platform.machine()}")

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
            "batch_size": self.batch_size,
            "dimension": self.dimension,
            "torch_version": self.torch_version,
            "platform": self.platform,
        }


def chunk_parts(parts: list[DocumentPart], url: str, title: str) -> list[Chunk]:
    """locator 를 유지한 채 자른다. 근거 검증이 위치를 특정할 수 있어야 하기 때문이다."""
    chunks: list[Chunk] = []
    for part in parts:
        text = normalize_text(part.text)
        if len(text) <= CHUNK_CHARS:
            chunks.append(
                Chunk(text=text, url=url, title=title, locator=part.locator, chunk_index=len(chunks))
            )
            continue
        step = CHUNK_CHARS - CHUNK_OVERLAP
        for start in range(0, len(text), step):
            piece = text[start : start + CHUNK_CHARS]
            if len(piece) < 100:  # 꼬리 조각은 앞 청크에 이미 포함돼 있다
                break
            chunks.append(
                Chunk(
                    text=piece,
                    url=url,
                    title=title,
                    locator=f"{part.locator}+{start}",
                    chunk_index=len(chunks),
                )
            )
    return chunks


class BM25Index:
    """키워드 색인. 임베딩 없이 동작하므로 오프라인·저비용 경로이기도 하다."""

    name = "bm25"

    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self._bm25 = BM25Okapi([self._tokenize(c.text) for c in chunks]) if chunks else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # 한글·영문·숫자를 함께 다뤄야 하므로 소문자화 후 공백 분할로 단순하게 간다.
        return normalize_text(text).lower().split()

    def rank(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(self._tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]


class DenseIndex:
    """오픈소스 임베딩 기반 의미 검색."""

    name = "dense"

    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str,
        batch_size: int = 16,
        normalize: bool = True,
    ):
        from langchain_huggingface import HuggingFaceEmbeddings

        self.chunks = chunks
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"batch_size": batch_size, "normalize_embeddings": normalize},
        )
        self._store = None
        self.run_info = EmbeddingRunInfo(
            model_name=model_name,
            device=self._resolve_device(),
            normalize_embeddings=normalize,
            batch_size=batch_size,
            torch_version=self._torch_version(),
        )
        if chunks:
            from langchain_community.vectorstores import FAISS

            self._store = FAISS.from_texts(
                [c.text for c in chunks],
                self.embeddings,
                metadatas=[{"chunk_index": c.chunk_index} for c in chunks],
            )
            self.run_info.dimension = len(self.embeddings.embed_query("dimension probe"))

    def _resolve_device(self) -> str:
        client = getattr(self.embeddings, "_client", None) or getattr(self.embeddings, "client", None)
        device = getattr(client, "device", None)
        return str(device) if device is not None else "unknown"

    @staticmethod
    def _torch_version() -> str | None:
        try:
            import torch

            return torch.__version__
        except ImportError:
            return None

    def rank(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if self._store is None:
            return []
        hits = self._store.similarity_search_with_score(query, k=k)
        out = []
        for doc, score in hits:
            index = doc.metadata.get("chunk_index")
            if index is not None and 0 <= index < len(self.chunks):
                out.append((self.chunks[index], float(score)))
        return out


class HybridIndex:
    """BM25 와 dense 결과를 RRF 로 융합한다."""

    name = "hybrid"

    def __init__(self, bm25: BM25Index, dense: DenseIndex | None):
        self.bm25 = bm25
        self.dense = dense

    def rank(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        pools = [self.bm25.rank(query, k * 2)]
        if self.dense is not None:
            pools.append(self.dense.rank(query, k * 2))

        fused: dict[int, float] = {}
        holder: dict[int, Chunk] = {}
        for pool in pools:
            for rank_position, (chunk, _score) in enumerate(pool, start=1):
                fused[chunk.chunk_index] = fused.get(chunk.chunk_index, 0.0) + 1.0 / (
                    RRF_K + rank_position
                )
                holder[chunk.chunk_index] = chunk
        order = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [(holder[idx], score) for idx, score in order]


def build_index(
    chunks: list[Chunk], *, embedding_model: str | None, batch_size: int = 16
) -> HybridIndex:
    """embedding_model 이 None 이면 BM25 단독으로 동작한다(오프라인·저비용 경로)."""
    bm25 = BM25Index(chunks)
    dense = (
        DenseIndex(chunks, model_name=embedding_model, batch_size=batch_size)
        if embedding_model and chunks
        else None
    )
    return HybridIndex(bm25, dense)
