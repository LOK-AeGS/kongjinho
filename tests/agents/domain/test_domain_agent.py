"""도메인 평가 에이전트 계약 테스트. LLM·네트워크 없이 순수 로직만 검사한다.

v3(단일 프롬프트 품질 검증)에서는 guard/linter/judge 코드가 없다. 여기서 검사하는 것은
- 참조 무결성(존재하지 않는 근거·주장을 조용히 걸러내기, 안 하면 KeyError로 죽는다)
- AppState 팀 스키마로의 변환(Evidence/Claim/VerdictRecord/Gap 필드 매핑)
- 관점 입력 격리
이며, "근거가 실제로 맞는 소리인가" 같은 품질 판단은 이제 모델의 self_check가 맡으므로
여기서 검사하지 않는다.

실행: python tests/agents/domain/test_domain_agent.py   (pytest 로도 동작)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agents.domain.node import (  # noqa: E402
    _to_team_claim, _to_team_evidence, _to_team_gaps, _to_team_record,
    project_input, summarize_technical,
)
from agents.domain.subgraph import DomainAnalysis, DraftClaim, DraftRecord, SelfCheck, _shape  # noqa: E402
from agents.domain.tools.evidence import (  # noqa: E402
    Evidence, make_evidence_id, merge_evidence, normalize_url,
)
from agents.domain.tools.websearch import classify_source  # noqa: E402
from scripts.run_domain import render_markdown  # noqa: E402

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
    assert twice[full["evidence_id"]]["published_date"] == "2024-06"


# --- 출처 정책 --------------------------------------------------------------

def test_신뢰_도메인만_등급을_받는다():
    assert classify_source("https://arxiv.org/abs/1234") == "paper"
    assert classify_source("https://www.nvidia.com/blog/x") == "vendor"
    assert classify_source("https://computeexpresslink.org/spec") == "standard"
    assert classify_source("https://medium.com/@someone/post") is None
    assert classify_source("https://youtube.com/watch?v=1") is None


def test_벤더_포럼은_공식_문서로_보지_않는다():
    assert classify_source("https://developer.nvidia.com/blog/x") == "vendor"
    assert classify_source("https://forums.developer.nvidia.com/t/x") is None
    assert classify_source("https://community.intel.com/t5/x") is None


# --- 참조 무결성 (v3에서 코드에 남긴 유일한 검증) ---------------------------

def _analysis(**kw):
    defaults = dict(
        claim_key="sw-mem-1", claim_tech=["sw"], basis="direct",
        evidence_ids=["E1"], text="주장", criterion="HBM 용량 압박 완화",
        rec_tech="sw", assessment="suitable", claim_keys=["sw-mem-1"],
    )
    defaults.update(kw)
    return DomainAnalysis(
        claims=[DraftClaim(
            claim_key=defaults["claim_key"], technology_ids=defaults["claim_tech"],
            text=defaults["text"], basis=defaults["basis"],
            evidence_ids=defaults["evidence_ids"], conditions=[], limitations=[],
        )],
        records=[DraftRecord(
            technology_id=defaults["rec_tech"], criterion=defaults["criterion"],
            basis=defaults["basis"], evidence_level="unknown", scope="direct",
            assessment=defaults["assessment"], value=None, findings="근거 요약",
            limitations=[], claim_keys=defaults["claim_keys"],
        )],
        self_check=SelfCheck(
            status="passed", violations=[], warnings=[],
            covered_axes=[defaults["criterion"]], missing_axes=[],
        ),
    )


def test_없는_라벨을_인용하면_조용히_걸러내고_공백으로_남긴다():
    ev = _evidence()
    claims, _, gaps = _shape(
        _analysis(evidence_ids=["E99"]), _store(ev), label_to_id={"E1": ev["evidence_id"]}
    )
    assert claims[0]["evidence_ids"] == []  # 존재하지 않는 라벨은 제거된다
    assert any("존재하지 않는 근거" in g for g in gaps)


def test_존재하는_라벨은_실제_ID로_환산된다():
    ev = _evidence()
    claims, _, _ = _shape(
        _analysis(evidence_ids=["E1"]), _store(ev), label_to_id={"E1": ev["evidence_id"]}
    )
    assert claims[0]["evidence_ids"] == [ev["evidence_id"]]


def test_실제_ID를_그대로_적어도_받아들인다():
    """모델이 라벨 대신 evidence_id 를 그대로 적는 경우도 있어 둘 다 받아준다."""
    ev = _evidence()
    claims, _, _ = _shape(
        _analysis(evidence_ids=[ev["evidence_id"]]), _store(ev), label_to_id={}
    )
    assert claims[0]["evidence_ids"] == [ev["evidence_id"]]


def test_기술이_어긋난_참조는_연결을_끊는다():
    """실측에서 sw 판정이 hw 주장을 가리키는 교차 참조가 나온 적이 있다."""
    ev = _evidence()
    _, records, gaps = _shape(
        _analysis(claim_tech=["hw"], rec_tech="sw", evidence_ids=["E1"], claim_keys=["sw-mem-1"]),
        _store(ev), label_to_id={"E1": ev["evidence_id"]},
    )
    assert records[0]["claim_ids"] == []
    assert records[0]["assessment"] == "unknown"
    assert any("다른 기술 주장" in g for g in gaps)


def test_주장은_한_문장_길이로_잘린다():
    ev = _evidence()
    claims, _, _ = _shape(
        _analysis(text="가" * 400, evidence_ids=["E1"]),
        _store(ev), label_to_id={"E1": ev["evidence_id"]},
    )
    assert len(claims[0]["text"]) == 240


def test_레코드가_연결된_주장의_근거를_모은다():
    ev = _evidence()
    _, records, _ = _shape(
        _analysis(evidence_ids=["E1"]), _store(ev), label_to_id={"E1": ev["evidence_id"]}
    )
    assert records[0]["evidence_ids"] == [ev["evidence_id"]]


def test_판정_없는_기술은_공백으로_기록된다():
    ev = _evidence()
    _, _, gaps = _shape(
        _analysis(evidence_ids=["E1"]), _store(ev), label_to_id={"E1": ev["evidence_id"]}
    )
    assert any("hw" in g and "판정 레코드" in g for g in gaps)


# --- 팀 AppState 스키마 변환 (node.py) ---------------------------------------

def test_강한_출처는_primary_direct로_분류된다():
    ev = _evidence(source_type="paper")
    team = _to_team_evidence(ev, claim_id="domain:claim:001")
    assert team["primary_or_secondary"] == "primary"
    assert team["direct_or_proxy"] == "direct"
    assert team["id"] == ev["evidence_id"]
    assert team["page_or_locator"] == ev["locator"]
    assert team["perspective"] == "domain"


def test_약한_출처는_secondary_proxy로_분류된다():
    ev = _evidence(source_type="news")
    team = _to_team_evidence(ev, claim_id=None)
    assert team["primary_or_secondary"] == "secondary"
    assert team["direct_or_proxy"] == "proxy"
    assert team["claim_id"] == ""  # 인용한 주장이 없으면 빈 문자열


def test_단일_기술_주장은_그_기술로_변환된다():
    claim = {"claim_id": "domain:claim:001", "technology_ids": ["sw"], "text": "t",
             "evidence_ids": [], "conditions": [], "limitations": []}
    assert _to_team_claim(claim)["technology"] == "sw"


def test_복수_기술_주장은_both로_변환된다():
    claim = {"claim_id": "domain:claim:001", "technology_ids": ["sw", "hw"], "text": "t",
             "evidence_ids": [], "conditions": [], "limitations": []}
    assert _to_team_claim(claim)["technology"] == "both"


def test_레코드_변환은_criterion과_assessment를_보존한다():
    record = {"technology_id": "sw", "criterion": "HBM 용량", "basis": "direct",
              "evidence_level": "production", "scope": "direct", "assessment": "suitable",
              "assessment_vocab": "domain_fit_v1", "value": "93.3%", "findings": "요약",
              "limitations": [], "evidence_ids": ["e1"]}
    team = _to_team_record(record)
    assert team["criterion"] == "HBM 용량"
    assert team["assessment"] == "suitable"
    assert team["perspective"] == "domain"
    assert team["stance_counts"] == {}


def test_missing_axes와_code_gaps가_모두_Gap으로_변환된다():
    gaps = _to_team_gaps(
        missing_axes=["전력·발열"],
        code_gaps=["전력·발열", "sw 기술에 대한 판정 레코드를 생성하지 못함"],
    )
    assert len(gaps) == 2
    assert gaps[0]["criterion"] == "전력·발열"
    assert gaps[1]["reason"] == "sw 기술에 대한 판정 레코드를 생성하지 못함"
    assert all(g["perspective"] == "domain" for g in gaps)


def test_마크다운_보고서에_레코드_근거가_표시된다():
    findings = {
        "status": "partial",
        "claims": [],
        "records": [{
            "technology": "sw", "criterion": "HBM 용량", "assessment": "suitable",
            "value": "93.3%", "findings": "KV cache 감소", "limitations": [],
            "evidence_ids": ["e1"],
        }],
        "gaps": [],
    }
    evidence = {
        "e1": {
            "title": "DeepSeek-V2", "url": "https://arxiv.org/abs/2405.04434",
            "page_or_locator": "p.21",
        }
    }
    report = render_markdown(findings, evidence, {"as_of": AS_OF})
    assert "[DeepSeek-V2](https://arxiv.org/abs/2405.04434) · p.21" in report


# --- 관점 입력 격리 (AppState 기준) -----------------------------------------

def test_다른_관점_결론은_입력에_섞이지_않는다():
    state = {
        "selected_tech": {"sw": {"name": "MLA"}, "hw": {"name": "ITME"}},
        "request": {"as_of": AS_OF, "max_search_rounds": 2},
        "technical_findings": {"claims": [], "records": []},
        "market_findings": {"claims": [{"text": "시장이 크다"}]},
        "stakeholder_findings": {"claims": [{"text": "업계가 반긴다"}]},
    }
    projected = project_input(state)
    blob = str(projected)
    assert "시장이 크다" not in blob and "업계가 반긴다" not in blob
    assert projected["sw_name"] == "MLA"
    assert projected["as_of_date"] == AS_OF


def test_selected_tech_키가_없으면_바로_에러난다():
    """ISSUE.md 1-2 가 지적한 문제: request.sw 를 읽으면 AppState 에선 KeyError."""
    state = {"selected_tech": {"sw": {"name": "MLA"}, "hw": {"name": "ITME"}},
             "request": {"as_of": AS_OF, "max_search_rounds": 1}}
    projected = project_input(state)  # KeyError 없이 정상 동작해야 한다
    assert projected["hw_name"] == "ITME"


def test_기술조사_결과가_없어도_동작한다():
    assert isinstance(summarize_technical(None), str)


def test_기술조사_새_스키마의_claims를_읽는다():
    findings = {"claims": [{"technology": "sw", "text": "MLA는 KV를 압축한다"}], "records": []}
    summary = summarize_technical(findings)
    assert "MLA는 KV를 압축한다" in summary


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
