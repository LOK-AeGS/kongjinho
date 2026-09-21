"""도메인 평가 에이전트 계약 테스트. LLM·네트워크 없이 순수 로직만 검사한다.

여기서 검사하는 규칙이 깨지면 근거 없는 주장이 보고서에 실려도 아무도 모른다.

실행: python tests/agents/domain/test_domain_agent.py   (pytest 로도 동작)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agents.domain.node import project_input, summarize_technical  # noqa: E402
from agents.domain.subgraph import DomainAnalysis, DraftClaim, DraftFit, _shape  # noqa: E402
from agents.domain.state import merge_by_perspective, to_team_findings  # noqa: E402
from agents.domain.quality.guard import run_guard  # noqa: E402
from agents.domain.quality.linter import lint_claims  # noqa: E402
from agents.domain.rag.evidence import (  # noqa: E402
    Evidence, make_evidence_id, merge_evidence, normalize_url,
)
from agents.domain.rag.websearch import classify_source  # noqa: E402

AS_OF = "2026-09-22"


def _evidence(**overrides) -> dict:
    base = Evidence.build(
        source_type="paper",
        title="DeepSeek-V2",
        author_or_organization="arxiv.org",
        url="https://arxiv.org/pdf/2405.04434",
        quote="reduces the KV cache by 93.3%",
        locator="p.21",
        retrieved_via="fetched_document",
        published_date="2024-06",
        content_sha256="abc",
    ).to_dict()
    base.update(overrides)
    return base


def _store(*evidences) -> dict:
    return {e["evidence_id"]: e for e in evidences}


# --- 근거 ID와 병합 ---------------------------------------------------------

def test_같은_출처는_재시도해도_같은_ID를_받는다():
    first = make_evidence_id("https://arxiv.org/pdf/2405.04434", "p.21", "KV cache  93.3%")
    second = make_evidence_id("https://ARXIV.org/pdf/2405.04434/", "p.21", "KV cache 93.3%")
    assert first == second  # 대소문자·공백·후행 슬래시 차이는 같은 근거다


def test_추적_파라미터는_ID에_영향을_주지_않는다():
    assert normalize_url("https://www.nvidia.com/a?utm_source=x") == "https://nvidia.com/a"


def test_다른_인용문은_다른_ID를_받는다():
    a = make_evidence_id("https://arxiv.org/x", "p.1", "첫 번째 문장")
    b = make_evidence_id("https://arxiv.org/x", "p.1", "두 번째 문장")
    assert a != b  # 같은 페이지라도 인용이 다르면 별개 근거


def test_병합은_멱등하고_기존_값을_지우지_않는다():
    full = _evidence()
    partial = {**full, "published_date": None}
    once = merge_evidence({}, _store(full))
    twice = merge_evidence(once, _store(partial))
    assert len(twice) == 1
    # 재시도에서 값이 None 으로 되돌아가면 안 된다
    assert twice[full["evidence_id"]]["published_date"] == "2024-06"


# --- 출처 정책 --------------------------------------------------------------

def test_신뢰_도메인만_등급을_받는다():
    assert classify_source("https://arxiv.org/abs/1234") == "paper"
    assert classify_source("https://www.nvidia.com/blog/x") == "vendor"
    assert classify_source("https://computeexpresslink.org/spec") == "standard"
    assert classify_source("https://medium.com/@someone/post") is None
    assert classify_source("https://youtube.com/watch?v=1") is None


def test_벤더_포럼은_공식_문서로_보지_않는다():
    """실측에서 forums.developer.nvidia.com 이 vendor 로 분류돼 근거가 과대평가됐다."""
    assert classify_source("https://developer.nvidia.com/blog/x") == "vendor"
    assert classify_source("https://forums.developer.nvidia.com/t/x") is None
    assert classify_source("https://community.intel.com/t5/x") is None


# --- 결정적 guard -----------------------------------------------------------

def test_원문에_없는_인용은_위반으로_잡는다():
    ev = _evidence(quote="존재하지 않는 문장입니다")
    report = run_guard(
        claims=[], evidence_store=_store(ev),
        source_texts={ev["url"]: "reduces the KV cache by 93.3%"}, as_of_date=AS_OF,
    )
    assert any(v.check == "quote" for v in report.violations)


def test_원문_스냅샷이_없으면_통과시키지_않는다():
    ev = _evidence()
    report = run_guard(claims=[], evidence_store=_store(ev), source_texts={}, as_of_date=AS_OF)
    assert any(v.check == "quote" for v in report.violations)


def test_locator가_비면_위반이다():
    ev = _evidence(locator="")
    report = run_guard(
        claims=[], evidence_store=_store(ev),
        source_texts={ev["url"]: ev["quote"]}, as_of_date=AS_OF,
    )
    assert any(v.check == "locator" for v in report.violations)


def test_기준일_이후_발행일은_위반이다():
    ev = _evidence(published_date="2027-01")
    report = run_guard(
        claims=[], evidence_store=_store(ev),
        source_texts={ev["url"]: ev["quote"]}, as_of_date=AS_OF,
    )
    assert any(v.check == "date" for v in report.violations)


def test_인용문에_없는_수치를_쓰면_위반이다():
    ev = _evidence()
    claim = {
        "claim_id": "domain:claim:001", "basis": "direct_evidence",
        "evidence_ids": [ev["evidence_id"]],
        "statement": "KV 캐시를 77.7% 줄인다",  # 인용문에는 93.3 만 있다
    }
    report = run_guard(
        claims=[claim], evidence_store=_store(ev),
        source_texts={ev["url"]: ev["quote"]}, as_of_date=AS_OF,
    )
    assert any(v.check == "numeric_unit" for v in report.violations)


def test_근거를_인용한_수치는_통과한다():
    ev = _evidence()
    claim = {
        "claim_id": "domain:claim:001", "basis": "direct_evidence",
        "evidence_ids": [ev["evidence_id"]],
        "statement": "KV 캐시를 93.3% 줄인다",
    }
    report = run_guard(
        claims=[claim], evidence_store=_store(ev),
        source_texts={ev["url"]: ev["quote"]}, as_of_date=AS_OF,
    )
    assert report.passed, report.to_dict()


def test_제품명_속_숫자는_측정값으로_보지_않는다():
    """실측에서 'DeepSeek-V2' 의 2 를 측정값으로 잡아 멀쩡한 주장을 강등시켰다."""
    ev = _evidence()
    claim = {
        "claim_id": "domain:claim:001", "basis": "direct_evidence",
        "evidence_ids": [ev["evidence_id"]],
        "statement": "DeepSeek-V2 는 KV 캐시를 93.3% 줄인다",  # 2 는 제품명, 93.3% 는 측정값
    }
    report = run_guard(
        claims=[claim], evidence_store=_store(ev),
        source_texts={ev["url"]: ev["quote"]}, as_of_date=AS_OF,
    )
    assert report.passed, report.to_dict()


def test_없는_근거를_참조하면_무결성_위반이다():
    claim = {
        "claim_id": "domain:claim:001", "basis": "direct_evidence",
        "evidence_ids": ["domain:ev:deadbeef"], "statement": "주장",
    }
    report = run_guard(claims=[claim], evidence_store={}, source_texts={}, as_of_date=AS_OF)
    assert any(v.check == "reference_integrity" for v in report.violations)


# --- 표현 린터 --------------------------------------------------------------

def test_근거_없는_승자_표현은_차단한다():
    claims = [{
        "claim_id": "domain:claim:001", "basis": "inference", "evidence_ids": [],
        "statement": "ITME가 MLA보다 더 우수하다",
    }]
    report = lint_claims(claims)
    assert "domain:claim:001" in report.blocking_ids


def test_추천_표현은_근거가_있어도_차단한다():
    claims = [{
        "claim_id": "domain:claim:001", "basis": "direct_evidence", "evidence_ids": ["e1"],
        "statement": "데이터센터에는 MLA를 채택해야 한다",
    }]
    assert "domain:claim:001" in lint_claims(claims).blocking_ids


def test_수치_비교_사실은_통과시킨다():
    claims = [{
        "claim_id": "domain:claim:001", "basis": "direct_evidence", "evidence_ids": ["e1"],
        "statement": "MLA는 KV 캐시 93.3% 감소, ITME는 처리량 1.80배 향상을 보고했다",
    }]
    assert not lint_claims(claims).blocking_ids  # 비교 사실 자체는 막지 않는다


# --- 주장 정형화 ------------------------------------------------------------

def _analysis(**kw):
    defaults = dict(
        claim_key="sw-mem-1", claim_tech=["sw"], basis="direct_evidence",
        evidence_ids=[], statement="주장", fit_tech="sw",
        assessment="suitable", claim_keys=["sw-mem-1"],
    )
    defaults.update(kw)
    return DomainAnalysis(
        claims=[DraftClaim(
            claim_key=defaults["claim_key"], technology_ids=defaults["claim_tech"],
            topic="메모리", statement=defaults["statement"], basis=defaults["basis"],
            evidence_ids=defaults["evidence_ids"], conditions=[], uncertainty="",
        )],
        fits=[DraftFit(
            technology_id=defaults["fit_tech"], requirements=["HBM"],
            assessment=defaults["assessment"], claim_keys=defaults["claim_keys"],
            limitations=[],
        )],
    )


def test_없는_근거_인용은_판단보류로_내린다():
    claims, _, gaps = _shape(_analysis(evidence_ids=["domain:ev:zzz"]), _store(_evidence()), [])
    assert claims[0]["basis"] == "unknown"
    assert claims[0]["uncertainty"]
    assert gaps


def test_매체_보도만_근거면_추론으로_내린다():
    news = _evidence(source_type="news", url="https://reuters.com/a")
    claims, _, _ = _shape(
        _analysis(evidence_ids=[news["evidence_id"]]), _store(news), ["연구용 셋업"]
    )
    assert claims[0]["basis"] == "inference"
    assert claims[0]["conditions"] == ["연구용 셋업"]


def test_기술이_어긋난_참조는_끊는다():
    ev = _evidence()
    _, fits, gaps = _shape(
        _analysis(claim_tech=["hw"], fit_tech="sw", evidence_ids=[ev["evidence_id"]]),
        _store(ev), [],
    )
    assert fits[0]["claim_ids"] == []
    assert fits[0]["assessment"] == "unknown"
    assert any("다른 기술 주장" in g for g in gaps)


def test_주장은_한_문장_길이로_잘린다():
    ev = _evidence()
    claims, _, _ = _shape(
        _analysis(statement="가" * 400, evidence_ids=[ev["evidence_id"]]), _store(ev), []
    )
    assert len(claims[0]["statement"]) == 240


# --- 관점 격리와 부모 State -------------------------------------------------

def test_다른_관점_결론은_입력에_섞이지_않는다():
    state = {
        "request": {
            "sw": {"name": "MLA"}, "hw": {"name": "ITME"},
            "as_of_date": AS_OF, "max_search_rounds": 2,
        },
        "technical_findings": {"claims": [], "trl_estimates": []},
        "market_findings": {"claims": [{"statement": "시장이 크다"}]},
        "stakeholder_findings": {"claims": [{"statement": "업계가 반긴다"}]},
    }
    projected = project_input(state)
    blob = str(projected)
    assert "시장이 크다" not in blob and "업계가 반긴다" not in blob


def test_관점별_품질은_키가_충돌하지_않는다():
    merged = merge_by_perspective({"market": {"guard": "ok"}}, {"domain": {"guard": "ok"}})
    assert set(merged) == {"market", "domain"}


def test_팀_스키마로_되돌릴_때_인용한_근거만_담는다():
    used, unused = _evidence(), _evidence(quote="다른 인용문", locator="p.9")
    findings = {
        "claims": [], "fits": [],
        "cited_evidence_ids": [used["evidence_id"]],
        "completion": {"status": "partial", "search_rounds_used": 1,
                       "revision_rounds_used": 0, "pages_used": 30, "gaps": [], "errors": []},
    }
    team = to_team_findings(findings, _store(used, unused))
    assert [e["evidence_id"] for e in team["evidence"]] == [used["evidence_id"]]
    assert "pages_used" not in team["completion"]  # 팀 AgentCompletion 에 없는 필드


def test_기술조사_결과가_없어도_동작한다():
    assert isinstance(summarize_technical(None), str)


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
