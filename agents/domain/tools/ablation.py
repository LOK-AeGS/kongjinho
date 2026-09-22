"""검색 방식 비교: BM25 / dense / hybrid.

같은 청크 집합과 같은 Golden Set에 대해 세 방식을 돌려 Hit@k, MRR, 지연, 메모리를 잰다.
측정을 같은 프로세스·같은 입력에서 하지 않으면 수치를 비교할 수 없다.

실행: python -m agents.domain.tools.ablation --cache data/fetch_cache --embedding BAAI/bge-m3
(ISSUE.md 5-2: --corpus 옵션은 없다. 문서 캐시 위치는 --cache, 기본값은 data/fetch_cache다.)
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

from agents.domain.tools.golden_set import GOLDEN_SET, GoldenItem, marker_hit
from agents.domain.tools.evidence import normalize_text
from agents.domain.tools.fetch import fetch_document
from agents.domain.tools.index import BM25Index, Chunk, DenseIndex, HybridIndex, chunk_parts

DEFAULT_SOURCES = {
    "sw-deepseek-v2": "https://arxiv.org/pdf/2405.04434",
    "hw-itme-cxl": "https://arxiv.org/pdf/2606.12556",
}


@dataclass
class MethodScore:
    method: str
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    mean_latency_ms: float
    p95_latency_ms: float
    peak_memory_mb: float
    queries: int


def _reciprocal_rank(ranked: list[Chunk], item: GoldenItem) -> float:
    for position, chunk in enumerate(ranked, start=1):
        if marker_hit(chunk.text, item):
            return 1.0 / position
    return 0.0


def evaluate(index, method: str, k: int = 5) -> MethodScore:
    latencies: list[float] = []
    hits = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks: list[float] = []

    tracemalloc.start()
    for item in GOLDEN_SET:
        started = time.perf_counter()
        ranked = [chunk for chunk, _ in index.rank(item.query, k)]
        latencies.append((time.perf_counter() - started) * 1000)

        for cutoff in hits:
            if any(marker_hit(c.text, item) for c in ranked[:cutoff]):
                hits[cutoff] += 1
        reciprocal_ranks.append(_reciprocal_rank(ranked, item))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total = len(GOLDEN_SET)
    ordered = sorted(latencies)
    return MethodScore(
        method=method,
        hit_at_1=round(hits[1] / total, 3),
        hit_at_3=round(hits[3] / total, 3),
        hit_at_5=round(hits[5] / total, 3),
        mrr=round(statistics.fmean(reciprocal_ranks), 3),
        mean_latency_ms=round(statistics.fmean(latencies), 1),
        p95_latency_ms=round(ordered[int(len(ordered) * 0.95) - 1], 1),
        peak_memory_mb=round(peak / 1024**2, 1),
        queries=total,
    )


def load_chunks(cache_dir: Path, sources: dict[str, str] | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc_id, url in (sources or DEFAULT_SOURCES).items():
        document = fetch_document(url, cache_dir=cache_dir)
        if document.error:
            raise RuntimeError(f"{doc_id} 수집 실패: {document.error}")
        chunks.extend(chunk_parts(document.parts, url, document.title or doc_id))
    # chunk_index 가 색인 내 위치를 가리켜야 하므로 통합 후 다시 매긴다.
    for position, chunk in enumerate(chunks):
        chunk.chunk_index = position
    return chunks


def run_ablation(
    cache_dir: Path,
    embedding_model: str | None,
    batch_size: int = 16,
    artifact_dir: Path | None = None,
) -> dict:
    chunks = load_chunks(cache_dir)
    bm25 = BM25Index(chunks)

    results = [asdict(evaluate(bm25, "bm25"))]
    embedding_info = None
    dense = None
    if embedding_model:
        dense = DenseIndex(chunks, model_name=embedding_model, batch_size=batch_size)
        embedding_info = dense.run_info.to_dict()
        results.append(asdict(evaluate(dense, f"dense:{embedding_model}")))
        results.append(asdict(evaluate(HybridIndex(bm25, dense), "hybrid:bm25+dense")))

    # 인덱스를 메모리에만 두면 같은 수치를 다시 만들 수 없다. 체크섬과 함께 남긴다.
    artifacts = []
    if artifact_dir is not None:
        from agents.domain.tools.artifacts import describe_cache, save_bm25, save_chunks, save_faiss

        artifacts.append(save_chunks(chunks, artifact_dir).to_dict())
        artifacts.append(save_bm25(bm25, artifact_dir).to_dict())
        if dense is not None:
            saved = save_faiss(dense, artifact_dir)
            if saved:
                artifacts.append(saved.to_dict())
        artifacts.append(describe_cache(cache_dir, "fetch_cache").to_dict())

    return {
        "chunk_count": len(chunks),
        "avg_chunk_chars": round(
            statistics.fmean(len(normalize_text(c.text)) for c in chunks), 1
        ),
        "golden_set_size": len(GOLDEN_SET),
        "embedding": embedding_info,
        "results": results,
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="검색 방식 비교 (BM25 / dense / hybrid)")
    parser.add_argument("--cache", default="data/fetch_cache", help="원문 캐시 디렉터리")
    parser.add_argument("--embedding", default="BAAI/bge-m3", help="빈 문자열이면 BM25만 측정")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", default="outputs/domain/ablation.json")
    parser.add_argument("--artifacts", default="outputs/domain/index_artifacts",
                        help="청크·FAISS·BM25 저장 위치")
    args = parser.parse_args()

    report = run_ablation(
        Path(args.cache), args.embedding or None, args.batch_size,
        artifact_dir=Path(args.artifacts),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"청크 {report['chunk_count']}개 / 질의 {report['golden_set_size']}개")
    header = f"{'method':<28}{'Hit@1':>7}{'Hit@3':>7}{'Hit@5':>7}{'MRR':>7}{'mean ms':>9}{'p95 ms':>8}{'peak MB':>9}"
    print(header)
    print("-" * len(header))
    for row in report["results"]:
        print(
            f"{row['method']:<28}{row['hit_at_1']:>7}{row['hit_at_3']:>7}{row['hit_at_5']:>7}"
            f"{row['mrr']:>7}{row['mean_latency_ms']:>9}{row['p95_latency_ms']:>8}"
            f"{row['peak_memory_mb']:>9}"
        )
    print(f"\n저장: {out_path}")
    for a in report["artifacts"]:
        print(f"  {a['name']:<16} {a['sha256'][:16]}  {a['bytes']:>10,}B  {a['note']}")


if __name__ == "__main__":
    main()
