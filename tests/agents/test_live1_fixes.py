"""live 1차 실행(2026-10-07)에서 품질 평가가 잡은 결함의 회귀 테스트. API 키 없이 실행된다."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.domain.node import _bind_numeric_claims  # noqa: E402
from agents.quality.checks import bias_control  # noqa: E402
from agents.report.metrics import annotate_metrics, metric_violations  # noqa: E402
from graph.supervisor import state_summary  # noqa: E402
from main import initial_state  # noqa: E402


def test_annotate_metrics_appends_condition_only_when_cited_evidence_has_it():
    text = "- MLA는 KV 캐시를 93.3% 줄였다.\n| SW | suitable (5.76×) |"
    evidence = "DeepSeek-V2 reduces KV cache by 93.3% vs DeepSeek 67B; 5.76x on 8 H800 GPUs."
    annotated = annotate_metrics(text, lambda line: evidence)
    assert "DeepSeek 67B" in annotated.splitlines()[0]
    assert "8×H800" in annotated.splitlines()[1]
    assert not any(metric_violations(line) for line in annotated.splitlines())


def test_annotate_metrics_does_not_invent_conditions_absent_from_evidence():
    # 근거에 없는 조건을 붙이면 Groundedness를 해친다(이전 동작: 무조건 부착).
    text = "- ITME는 처리량을 35.7% 높였다."
    assert annotate_metrics(text, lambda line: "ITME achieves 1.80x over NVMe-oF.") == text
    assert annotate_metrics(text) == text


def test_annotate_metrics_skips_sentences_about_the_missing_condition():
    # live 3차: "비교 기준이 누락되어 …" 문장 끝에 그 기준을 붙여 뜻이 모순됐다.
    text = "- 5.76배 수치는 비교 기준이 누락되어 상충이 발생했다."
    assert annotate_metrics(text, lambda line: "8 H800 GPUs") == text


def test_annotate_metrics_keeps_lines_that_already_have_conditions():
    line = "- ITME는 CPU-offload 대비 최대 35.7% 처리량 향상을 보였다."
    assert annotate_metrics(line) == line


def test_domain_claim_with_number_missing_from_cited_quote_is_dropped_to_gap():
    # live 1차: "KV 캐시를 93.3% 줄인다"가 93.3이 없는 TransMLA 발췌문을 인용했다.
    claims = [
        {"claim_id": "domain:claim:001", "technology_ids": ["sw"], "text": "MLA는 KV 캐시를 93.3% 줄인다.",
         "evidence_ids": ["domain:ev:a"], "conditions": [], "limitations": []},
        {"claim_id": "domain:claim:002", "technology_ids": ["sw"], "text": "MLA는 KV 캐시를 93.3% 줄인다.",
         "evidence_ids": ["domain:ev:b"], "conditions": [], "limitations": []},
    ]
    records = [{"technology_id": "sw", "criterion": "HBM 용량 압박 완화", "claim_ids": ["domain:claim:001", "domain:claim:002"],
                "evidence_ids": ["domain:ev:a"], "value": "93.3%", "limitations": []}]
    store = {
        "domain:ev:a": {"quote": "MLA compresses the key-value cache using low-rank matrices."},
        "domain:ev:b": {"quote": "DeepSeek-V2 reduces the KV cache by 93.3% compared with DeepSeek 67B."},
    }
    kept, cleaned, gaps, limitations, dropped = _bind_numeric_claims(claims, records, store)
    assert [claim["claim_id"] for claim in kept] == ["domain:claim:002"]
    assert dropped == 1
    assert cleaned[0]["value"] is None
    assert cleaned[0]["claim_ids"] == ["domain:claim:002"]
    assert any(gap["missing_evidence"] == ["93.3"] for gap in gaps)
    assert limitations


def _one_technology_section(gaps):
    # 보고서 전체로는 SW·HW가 균형을 이루고(5.1), 4.3만 SW를 다루는 live 1차 상황을 재현한다.
    store = {
        f"e{index}": {"url": f"https://source{index}.example/report", "stance": "support"}
        for index in range(1, 7)
    }
    findings = {
        "stakeholder": {"records": [
            {"technology": "sw", "evidence_ids": [f"e{index}"]} for index in range(1, 4)
        ], "claims": [], "gaps": gaps},
        "technical": {"records": [
            {"technology": "hw", "evidence_ids": [f"e{index}"]} for index in range(4, 7)
        ], "claims": [], "gaps": []},
    }
    markdown = (
        "# SUMMARY\n\n## 5.1 비교 매트릭스\n\n"
        + "\n".join(f"- HW 기술 사실 {index}. 〔근거: e{index}〕" for index in range(4, 7))
        + "\n\n## 4.3 이해관계자\n\n"
        + "\n".join(f"- 이해관계자 발언 {index}. 〔근거: e{index}〕" for index in range(1, 4))
    )
    return bias_control(markdown, store, findings)


def test_one_technology_section_is_warning_when_other_technology_gap_is_documented():
    result = _one_technology_section([{"technology": "hw", "criterion": "투자·산업 관계자", "reason": "not_found"}])
    assert result["passed"]
    assert any("경고: 4.3 이해관계자: 한 기술만 다룸" in detail for detail in result["details"])


def test_one_technology_section_still_fails_without_documented_absence():
    result = _one_technology_section([])
    assert not result["passed"]
    assert "4.3 이해관계자: 한 기술만 다룸" in result["details"]


def test_supervisor_summary_carries_evidence_count_and_next_report_version():
    state = initial_state()
    state.update({
        "domain_findings": {"status": "partial", "records": [
            {"basis": "direct", "evidence_ids": [f"domain:e{index}"]} for index in range(16)
        ], "claims": [], "gaps": []},
        "evidence_store": {f"domain:e{index}": {} for index in range(16)},
        "node_status": {"domain": {"status": "done", "attempts": 1, "completed_step": 2, "sufficiency": None}},
        "report_version": 1,
    })
    summary = state_summary(state)
    assert summary["perspectives"]["domain"]["evidence_count"] == 16
    assert summary["next_report_is"] == "version 2"


def test_line_citing_evidence_without_its_number_is_blocking():
    # live 2차: LLM writer가 technical 주장의 93.3%를 수치가 없는 도메인 근거에 붙였다.
    from agents.report.validators import _citation_binding_issues

    context = {
        "evidence_store": {
            "domain:ev:x": {"excerpt": "MLA reduces the KV cache to 70KB per token in DeepSeek-V3."},
            "technical:ev:y": {"excerpt": "DeepSeek-V2 reduces the KV cache by 93.3% compared with DeepSeek 67B."},
        },
        "claims": {},
        "findings": {},
    }
    mixed = "- MLA는 DeepSeek 67B 대비 KV 캐시를 93.3% 줄인다. 〔근거: domain:ev:x〕"
    bound = "- MLA는 DeepSeek 67B 대비 KV 캐시를 93.3% 줄인다. 〔근거: technical:ev:y〕"
    assert [item["code"] for item in _citation_binding_issues("domain", mixed, context)] == ["numeric_citation_mismatch"]
    assert _citation_binding_issues("domain", bound, context) == []


def test_upstream_claim_breaking_report_rules_is_excluded_from_deterministic_render():
    # live 3차: 도메인 주장 "성능 우위", 시장 주장 "학습 비용 42.5% 절감(MLA 귀속)"이 대체 렌더에도 남았다.
    from agents.report.subgraph import _violates_report_rules

    assert _violates_report_rules("MLA는 기존 MHA 대비 성능 우위를 목표로 설계되었다.")
    assert _violates_report_rules("MLA로 학습 비용을 42.5% 절감했다.")
    assert not _violates_report_rules("ITME는 CPU-offload 대비 최대 35.7% 처리량 향상을 보였다.")


def test_coverage_counts_cited_line_that_also_mentions_unconfirmed_limits():
    # live 3차: TRL 줄이 '운영 사례 미확인' 한계를 함께 담아 technical 관점 누락으로 오판됐다.
    from agents.quality.checks import coverage

    sections = {
        "4.1 기술 성숙도(TRL)": "- HW: TRL 4–5 — laboratory validation / 상용 사례 미확인 〔근거: e1〕",
        "4.2 시장성": "- 시장 근거 〔근거: e2〕",
        "4.3 이해관계자": "- 발언 〔근거: e3〕",
        "4.4 도메인 적용": "- 판정 〔근거: e4〕",
    }
    markdown = "# SUMMARY\n\n" + "\n\n".join(f"## {title}\n\n{body}" for title, body in sections.items())
    assert coverage(markdown)["passed"]


def test_table_cell_drops_metric_whose_condition_is_not_in_evidence():
    # live 5차: 비교 매트릭스 셀 'suitable (93.3%)'가 조건 없이 실려 보고서 검증 위반으로 남았다.
    from agents.report.subgraph import _safe_cell

    context = {
        "evidence_store": {"e1": {"excerpt": "MLA compresses the KV cache."}},
        "claims": {},
        "findings": {},
    }
    assert _safe_cell("suitable (93.3%)", ["e1"], context) == "suitable"
    assert _safe_cell("supported", ["e1"], context) == "supported"
