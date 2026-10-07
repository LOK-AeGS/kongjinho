"""live 1차 실행(2026-10-07)에서 품질 평가가 잡은 결함의 회귀 테스트. API 키 없이 실행된다."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.domain.node import _bind_numeric_claims  # noqa: E402
from agents.quality.checks import bias_control  # noqa: E402
from agents.report.metrics import annotate_metrics, metric_violations  # noqa: E402
from graph.supervisor import state_summary  # noqa: E402
from main import initial_state  # noqa: E402


def test_annotate_metrics_appends_canonical_conditions_in_text_and_table_cells():
    text = "- MLA는 KV 캐시를 93.3% 줄였다.\n| SW | suitable (5.76×) |"
    annotated = annotate_metrics(text)
    assert "DeepSeek 67B" in annotated.splitlines()[0]
    assert "8×H800" in annotated.splitlines()[1]
    assert not any(metric_violations(line) for line in annotated.splitlines())


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
