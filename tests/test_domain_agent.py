"""도메인 평가 에이전트의 근거 검증 계약 테스트.

이 강등 규칙이 깨지면 근거 없는 주장이 direct_evidence 로 보고서에 실려도 아무도 모른다.
LLM 호출 없이 순수 함수만 검사하므로 API 키 없이 실행된다.

실행: python tests/test_domain_agent.py   (pytest 로도 동작)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.domain_agent import (  # noqa: E402
    DomainAnalysis,
    DraftClaim,
    DraftFit,
    SearchHit,
    _evidence_key,
    _validate,
    summarize_technical,
)

PAPER_EVIDENCE = {
    "evidence_id": "domain:ev:001",
    "document_id": "sw-deepseek-v2",
    "source_type": "paper",
    "title": "DeepSeek-V2",
    "author_or_organization": "DeepSeek-AI",
    "url": "https://arxiv.org/pdf/2405.04434",
    "published_date": "2024-06",
    "accessed_date": "2026-09-21",
    "page": 7,
    "section": None,
    "excerpt": "MLA reduces KV cache by 93.3%",
}
WEAK_EVIDENCE = {
    **PAPER_EVIDENCE,
    "evidence_id": "domain:ev:002",
    "document_id": None,
    "source_type": "community",
    "title": "forum post",
    "url": "https://example.com/post",
    "page": None,
    "excerpt": "누군가 좋다고 했다",
}
EVIDENCE = [PAPER_EVIDENCE, WEAK_EVIDENCE]


def _analysis(
    *, basis, evidence_ids, statement="주장", assessment="suitable",
    claim_keys=("sw-mem-1",), claim_tech=("sw",), fit_tech="sw",
):
    return DomainAnalysis(
        claims=[
            DraftClaim(
                claim_key="sw-mem-1",
                technology_ids=list(claim_tech),
                topic="메모리",
                statement=statement,
                basis=basis,
                evidence_ids=list(evidence_ids),
                conditions=[],
                uncertainty="",
            )
        ],
        fits=[
            DraftFit(
                technology_id=fit_tech,
                requirements=["HBM 용량"],
                assessment=assessment,
                claim_keys=list(claim_keys),
                limitations=[],
            )
        ],
    )


def test_없는_근거를_인용하면_판단보류로_내린다():
    claims, _, gaps = _validate(
        _analysis(basis="direct_evidence", evidence_ids=["domain:ev:999"]), EVIDENCE, []
    )
    assert claims[0]["basis"] == "unknown"
    assert claims[0]["evidence_ids"] == []
    assert claims[0]["uncertainty"]  # 비워두지 않는다
    assert gaps


def test_커뮤니티_출처만_있으면_추론으로_내린다():
    claims, _, _ = _validate(
        _analysis(basis="direct_evidence", evidence_ids=["domain:ev:002"]),
        EVIDENCE,
        ["연구용 셋업"],
    )
    assert claims[0]["basis"] == "inference"
    # 추론은 전제를 반드시 남긴다
    assert claims[0]["conditions"] == ["연구용 셋업"]


def test_논문_근거는_사실주장으로_유지된다():
    claims, fits, _ = _validate(
        _analysis(
            basis="direct_evidence", evidence_ids=["domain:ev:001"], assessment="conditional"
        ),
        EVIDENCE,
        [],
    )
    assert claims[0]["basis"] == "direct_evidence"
    assert fits[0]["assessment"] == "conditional"
    assert fits[0]["claim_ids"] == ["domain:claim:001"]


def test_주장은_한_문장_길이로_잘린다():
    claims, _, _ = _validate(
        _analysis(basis="direct_evidence", evidence_ids=["domain:ev:001"], statement="가" * 400),
        EVIDENCE,
        [],
    )
    assert len(claims[0]["statement"]) == 240


def test_뒷받침_주장이_없는_판정은_보류로_내린다():
    analysis = _analysis(basis="direct_evidence", evidence_ids=["domain:ev:001"])
    analysis.claims = []  # 주장이 사라진 상태
    _, fits, gaps = _validate(analysis, EVIDENCE, [])
    assert fits[0]["assessment"] == "unknown"
    assert fits[0]["claim_ids"] == []
    assert gaps


def test_평가되지_않은_기술은_공백으로_기록된다():
    _, _, gaps = _validate(
        _analysis(basis="direct_evidence", evidence_ids=["domain:ev:001"]), EVIDENCE, []
    )
    assert any("hw" in g for g in gaps)


def test_같은_페이지_다른_청크는_중복으로_보지_않는다():
    """URL·페이지만으로 중복 판정하면 같은 페이지의 다른 근거가 소실된다."""
    common = {
        "title": "DeepSeek-V2",
        "url": "https://arxiv.org/pdf/2405.04434",
        "source_type": "paper",
        "author_or_organization": "DeepSeek-AI",
        "page": 7,
    }
    first = SearchHit(excerpt="MLA compresses KV into latent vectors", **common)
    second = SearchHit(excerpt="Throughput improves by 5.76x on 236B MoE", **common)
    assert _evidence_key(first) != _evidence_key(second)
    assert _evidence_key(first) == _evidence_key(SearchHit(excerpt=first.excerpt, **common))


def test_기술이_어긋난_참조는_끊고_공백으로_남긴다():
    """sw 판정이 hw 주장을 가리키는 교차 참조가 실제 실행에서 나왔다."""
    analysis = _analysis(
        basis="direct_evidence",
        evidence_ids=["domain:ev:001"],
        claim_tech=("hw",),   # 주장은 hw 것인데
        fit_tech="sw",        # 판정은 sw
    )
    _, fits, gaps = _validate(analysis, EVIDENCE, [])
    assert fits[0]["claim_ids"] == []
    assert fits[0]["assessment"] == "unknown"
    assert any("다른 기술의 주장" in g for g in gaps)


def test_기술조사_결과가_없어도_동작한다():
    assert isinstance(summarize_technical({"technical_findings": None}), str)


if __name__ == "__main__":
    failures = []
    for name, func in sorted(globals().items()):
        if not name.startswith("test_") or not callable(func):
            continue
        try:
            func()
            print(f"  OK   {name}")
        except AssertionError as exc:
            failures.append(name)
            print(f"  FAIL {name}: {exc}")
    print("\n" + ("전체 통과" if not failures else f"실패 {len(failures)}건: {failures}"))
    sys.exit(1 if failures else 0)
