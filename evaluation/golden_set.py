"""검색 품질 측정용 Golden Set.

정답을 "특정 청크 ID"가 아니라 "그 청크에 반드시 들어 있어야 하는 표지 문자열"로 정의한다.
청킹 파라미터를 바꾸면 청크 경계가 달라져 ID 기반 정답은 전부 무효가 되지만,
표지 기반이면 파라미터를 바꿔도 같은 기준으로 비교할 수 있다.

한국어 질의로 영어 원문을 찾는 교차언어 상황을 일부러 섞었다.
도메인 평가 질문이 한국어로 생성되기 때문이다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenItem:
    query: str
    # 검색된 청크 안에 하나라도 있으면 적중으로 본다(대소문자 무시).
    markers: tuple[str, ...]
    note: str = ""


# 대상 문서: DeepSeek-V2(arXiv 2405.04434), ITME(arXiv 2606.12556)
GOLDEN_SET: tuple[GoldenItem, ...] = (
    GoldenItem("MLA는 KV 캐시를 얼마나 줄이는가", ("93.3",), "핵심 수치, 교차언어"),
    GoldenItem("KV cache reduction ratio of Multi-head Latent Attention", ("93.3",), "영문 동일 질의"),
    GoldenItem("DeepSeek-V2의 전체 파라미터 수와 활성 파라미터 수", ("236B", "21B"), "모델 규모"),
    GoldenItem("maximum generation throughput improvement", ("5.76",), "처리량 배수"),
    GoldenItem("지원하는 최대 문맥 길이는", ("128K",), "문맥 길이"),
    GoldenItem("training cost saving compared to DeepSeek 67B", ("42.5",), "학습 비용"),
    GoldenItem("Multi-head Latent Attention low-rank key value joint compression", ("low-rank",), "기법 원리"),
    GoldenItem("MoE 아키텍처에서 전문가 라우팅 방식", ("expert",), "구조"),
    GoldenItem("ITME가 사용하는 메모리 계층 구조", ("CXL",), "HW 핵심"),
    GoldenItem("CXL hybrid memory tiering for LLM inference", ("CXL",), "영문 동일 질의"),
    GoldenItem("ITME throughput improvement over baseline", ("1.80", "1.8"), "HW 처리량"),
    GoldenItem("어떤 데이터를 GPU 메모리에 유지하는가", ("GPU Memory", "T1"), "계층 배치"),
    GoldenItem("disaggregated memory latency overhead", ("latency",), "지연 이슈"),
    GoldenItem("KV cache offloading to slower memory tier", ("KV",), "오프로딩"),
    GoldenItem("attention 연산에서 key value 캐시가 차지하는 메모리", ("KV cache", "KV"), "문제 정의"),
    GoldenItem("inference batch size and sequence length in evaluation", ("batch",), "실험 조건"),
)


def marker_hit(text: str, item: GoldenItem) -> bool:
    lowered = (text or "").lower()
    return any(marker.lower() in lowered for marker in item.markers)
