"""고정 영문 질의 + BM25/dense/RRF 검색."""

from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Protocol

from .corpus import CorpusChunk


CRITERIA = (
    "mechanism",
    "application_scope",
    "performance",
    "limitations",
    "validation_environment",
)

QUERY_TEMPLATES: dict[str, dict[str, tuple[str, str]]] = {
    "sw": {
        "mechanism": (
            "DeepSeek-V2 MLA multi-head latent attention KV cache compression mechanism",
            "MLA low-rank latent KV projection decoupled RoPE architecture",
        ),
        "application_scope": (
            "DeepSeek-V2 MLA deployment requirements model architecture compatibility serving",
            "MLA inference implementation constraints training architecture integration scope",
        ),
        "performance": (
            "DeepSeek-V2 MLA KV cache reduction generation throughput benchmark conditions",
            "MLA 93.3 percent KV cache 5.76 times throughput H800 baseline",
        ),
        "limitations": (
            "DeepSeek-V2 MLA limitations quality tradeoff implementation complexity",
            "MLA constraints kernel serving compatibility accuracy evidence limitations",
        ),
        "validation_environment": (
            "DeepSeek-V2 MLA validation environment GPU model context batch production evidence",
            "MLA experiments 8 H800 serving workload deployment operational validation",
        ),
    },
    "hw": {
        "mechanism": (
            "ITME tiered memory expansion CXL hybrid memory KV cache mechanism RDMA SSD",
            "ITME CXL CMM byte addressable remote memory prefetch architecture",
        ),
        "application_scope": (
            "ITME deployment requirements vLLM GPU cluster CXL memory infrastructure",
            "ITME system integration CMM SSD RDMA serving compatibility scope",
        ),
        "performance": (
            "ITME throughput TTFT benchmark NVMe-oF recompute CPU offload conditions",
            "ITME 1.80 1.81 35.7 percent performance baseline turn benchmark",
        ),
        "limitations": (
            "ITME limitations CXL bandwidth latency prefetch failure infrastructure cost",
            "ITME prototype constraints contention scalability overhead blocking evidence",
        ),
        "validation_environment": (
            "ITME evaluation environment GPU CXL CMM SSD RDMA vLLM prototype scale",
            "ITME hardware testbed workload model context operational production evidence",
        ),
    },
}


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def select_query(technology: str, criterion: str, attempt: int) -> str:
    variants = QUERY_TEMPLATES[technology][criterion]
    return variants[min(max(attempt - 1, 0), len(variants) - 1)]


def select_device(explicit: str | None = None) -> str:
    requested = (explicit or os.getenv("EMBEDDING_DEVICE") or "auto").lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("EMBEDDING_DEVICE는 auto, cuda, mps, cpu 중 하나여야 합니다.")
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class Embedder(Protocol):
    model_name: str
    device: str

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class BGEEmbedder:
    """실행 시점에만 sentence-transformers와 모델을 로드한다."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        self.model_name = model_name
        self.device = select_device(device)
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - 설치 안내 경로
                raise RuntimeError(
                    "실제 dense 검색에는 sentence-transformers가 필요합니다."
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()


@dataclass(frozen=True)
class SearchHit:
    chunk: CorpusChunk
    score: float
    bm25_rank: int | None
    dense_rank: int | None
    query: str
    criterion: str

    def to_candidate(self) -> dict:
        value = self.chunk.to_dict()
        value.update(
            {
                "rrf_score": self.score,
                "bm25_rank": self.bm25_rank,
                "dense_rank": self.dense_rank,
                "queries": [self.query],
                "matched_criteria": [self.criterion],
            }
        )
        return value


def _bm25_scores(tokenized_docs: list[list[str]], query_tokens: list[str]) -> list[float]:
    """작은 고정 코퍼스용 BM25. 외부 패키지 없이 단위 테스트가 가능하다."""
    if not tokenized_docs:
        return []
    n_docs = len(tokenized_docs)
    avg_len = sum(len(doc) for doc in tokenized_docs) / max(n_docs, 1)
    doc_freq: dict[str, int] = {}
    for doc in tokenized_docs:
        for token in set(doc):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    scores: list[float] = []
    k1, b = 1.5, 0.75
    for doc in tokenized_docs:
        counts: dict[str, int] = {}
        for token in doc:
            counts[token] = counts.get(token, 0) + 1
        score = 0.0
        for token in query_tokens:
            freq = counts.get(token, 0)
            if not freq:
                continue
            n = doc_freq.get(token, 0)
            idf = math.log(1 + (n_docs - n + 0.5) / (n + 0.5))
            norm = freq + k1 * (1 - b + b * len(doc) / max(avg_len, 1))
            score += idf * freq * (k1 + 1) / norm
        scores.append(score)
    return scores


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(v * v for v in left))
    right_norm = math.sqrt(sum(v * v for v in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _ranks(scores: list[float], *, positive_only: bool = False) -> dict[int, int]:
    ordered = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    ranks: dict[int, int] = {}
    for index in ordered:
        if positive_only and scores[index] <= 0:
            continue
        ranks[index] = len(ranks) + 1
    return ranks


class HybridRetriever:
    """기술별로 격리한 BM25 + dense 순위를 RRF로 결합한다."""

    def __init__(
        self,
        chunks: list[CorpusChunk],
        *,
        embedder: Embedder | None = None,
        rrf_k: int = 60,
    ):
        self.chunks = list(chunks)
        self.embedder = embedder
        self.rrf_k = rrf_k
        self._dense_cache: dict[str, list[float]] | None = None

    @property
    def run_info(self) -> dict:
        return {
            "retrieval": "bm25+dense+rrf" if self.embedder else "bm25",
            "embedding_model": getattr(self.embedder, "model_name", None),
            "embedding_device": getattr(self.embedder, "device", None),
            "rrf_k": self.rrf_k,
        }

    def _dense_vectors(self) -> dict[str, list[float]]:
        if self.embedder is None:
            return {}
        if self._dense_cache is None:
            vectors = self.embedder.encode([chunk.text for chunk in self.chunks])
            self._dense_cache = {
                chunk.chunk_id: vector for chunk, vector in zip(self.chunks, vectors)
            }
        return self._dense_cache

    def search(
        self,
        technology: str,
        criterion: str,
        query: str,
        *,
        top_k: int = 4,
    ) -> list[SearchHit]:
        selected = [chunk for chunk in self.chunks if chunk.technology == technology]
        if not selected:
            return []
        bm25_values = _bm25_scores([tokenize(chunk.text) for chunk in selected], tokenize(query))
        bm25_ranks = _ranks(bm25_values, positive_only=True)

        dense_ranks: dict[int, int] = {}
        if self.embedder is not None:
            query_vector = self.embedder.encode([query])[0]
            cache = self._dense_vectors()
            dense_values = [_cosine(query_vector, cache[chunk.chunk_id]) for chunk in selected]
            dense_ranks = _ranks(dense_values)

        combined: list[tuple[int, float]] = []
        for index in range(len(selected)):
            score = 0.0
            if index in bm25_ranks:
                score += 1.0 / (self.rrf_k + bm25_ranks[index])
            if index in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[index])
            if score > 0:
                combined.append((index, score))
        combined.sort(key=lambda item: (-item[1], selected[item[0]].chunk_id))
        return [
            SearchHit(
                chunk=selected[index],
                score=score,
                bm25_rank=bm25_ranks.get(index),
                dense_rank=dense_ranks.get(index),
                query=query,
                criterion=criterion,
            )
            for index, score in combined[:top_k]
        ]


def merge_candidates(existing: list[dict], incoming: list[dict]) -> list[dict]:
    merged = {item["chunk_id"]: dict(item) for item in existing}
    for item in incoming:
        current = merged.get(item["chunk_id"])
        if current is None:
            merged[item["chunk_id"]] = dict(item)
            continue
        current["rrf_score"] = max(current.get("rrf_score", 0), item.get("rrf_score", 0))
        current["queries"] = list(dict.fromkeys(current.get("queries", []) + item.get("queries", [])))
        current["matched_criteria"] = list(
            dict.fromkeys(current.get("matched_criteria", []) + item.get("matched_criteria", []))
        )
    return sorted(merged.values(), key=lambda value: value["chunk_id"])


def coverage_by_technology(candidates: list[dict]) -> dict[str, dict[str, bool]]:
    coverage = {
        technology: {criterion: False for criterion in CRITERIA}
        for technology in ("sw", "hw")
    }
    for candidate in candidates:
        technology = candidate.get("technology")
        if technology not in coverage:
            continue
        for criterion in candidate.get("matched_criteria", []):
            if criterion in coverage[technology]:
                coverage[technology][criterion] = True
    return coverage


def missing_criteria(coverage: dict[str, dict[str, bool]]) -> list[str]:
    return [
        f"{technology}:{criterion}"
        for technology in ("sw", "hw")
        for criterion in CRITERIA
        if not coverage.get(technology, {}).get(criterion, False)
    ]

