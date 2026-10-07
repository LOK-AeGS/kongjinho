"""보고서 인용 정규화·payload 정리·한계점·SUMMARY 상한·품질 재작업 피드백."""
from agents.report.subgraph import normalize_draft, normalize_state, section_payload
from agents.report.validators import MAX_SUMMARY_CHARS, validate_section

EV = {"id": "technical:ev:aaa", "claim_id": "technical:claim:1", "doc_id": "arxiv:1.0v1", "title": "t", "author_or_org": "a",
      "source_type": "paper", "primary_or_secondary": "primary", "direct_or_proxy": "direct", "url": None, "published_at": "2026-01",
      "accessed_at": "2026-09-22", "page_or_locator": "p1", "quote": "q", "stance": "support", "evidence_level": "unknown",
      "metric_tag": None, "perspective": "technical", "content_hash": "h"}


def findings(status="complete", gaps=()):
    return {"perspective": "technical", "status": status, "records": [], "gaps": list(gaps), "limitations": [],
            "input_evidence_ids": ["technical:ev:aaa"], "meta": {"search_logs": ["noise"]},
            "claims": [{"claim_id": "technical:claim:1", "technology": "sw", "perspective": "technical", "text": "MLA 주장",
                        "evidence_ids": ["technical:ev:aaa"], "conditions": [], "limitations": []}]}


def state(**extra):
    base = {"request": {"as_of": "2026-09-22"}, "selected_tech": {"sw": {"name": "A"}, "hw": {"name": "B"}}, "domain": "datacenter",
            "technical_findings": findings(), "evidence_store": {"technical:ev:aaa": EV}, "retries": {"report": 2}}
    base.update(extra)
    return base


def draft(markdown, section_id="background", claim_ids=(), evidence_ids=()):
    return {"section_id": section_id, "title": "x", "markdown": markdown, "claim_ids": list(claim_ids), "evidence_ids": list(evidence_ids)}


def test_all_id_families_become_evidence_ids_and_unknown_are_dropped():
    ctx = normalize_state(state())
    out = normalize_draft(draft("가 〔근거: technical:claim:1〕 나 〔근거: arxiv:1.0v1〕 다 〔근거: technical:chunk:dead〕 라 〔근거: technical:ev:aaa〕"), ctx)
    assert out["markdown"].count("〔근거: technical:ev:aaa〕") == 3
    assert "chunk" not in out["markdown"] and "claim:1" not in out["markdown"] and "arxiv" not in out["markdown"]
    assert out["evidence_ids"] == ["technical:ev:aaa"] and out["claim_ids"] == ["technical:claim:1"]


def test_wrong_brackets_are_normalized_and_not_glued():
    ctx = normalize_state(state())
    out = normalize_draft(draft("끝났다【technical:ev:aaa】.\n\n〔근거: technical:ev:aaa】 그리고 【참고】"), ctx)
    assert "【technical" not in out["markdown"] and out["markdown"].count("〔근거: technical:ev:aaa〕") == 2
    assert "【참고】" in out["markdown"]  # 인용이 아닌 괄호는 그대로


def test_limitations_always_names_partial_and_failed_perspectives():
    ctx = normalize_state(state(technical_findings=findings(status="partial")))
    out = normalize_draft(draft("# 6. 한계점\n공개 정보의 한계가 있다.", "limitations"), ctx)
    assert "technical upstream 상태는 partial" in out["markdown"]
    again = normalize_draft(out, ctx)
    assert again["markdown"].count("technical upstream 상태는 partial") == 1


def test_payload_hides_internal_fields_and_rejected_chunk_ids():
    f = findings(gaps=[{"technology": "both", "perspective": "technical", "criterion": "evidence",
                        "reason": "원문에 없는 인용문: technical:chunk:6cb9feb6c5e20668", "missing_evidence": []}])
    ctx = normalize_state(state(technical_findings=f))
    payload = section_payload("background", ctx)
    text = str(payload)
    assert "chunk:6cb9" not in text and "search_logs" not in text and "input_evidence_ids" not in text
    assert "document_id" not in text and "arxiv:1.0v1" not in text
    assert "technical:ev:aaa" in payload["evidence"]


def test_summary_over_limit_is_a_blocking_issue():
    ctx = normalize_state(state())
    long = "# SUMMARY\n" + "가" * (MAX_SUMMARY_CHARS + 50)
    issues = validate_section(draft(long, "summary") | {"title": "SUMMARY"}, ctx)
    assert any(i["code"] == "summary_too_long" and i["blocking"] for i in issues)
    ok = validate_section(draft("# SUMMARY\n" + "가" * 200, "summary") | {"title": "SUMMARY"}, ctx)
    assert not any(i["code"] == "summary_too_long" for i in ok)


def test_quality_rework_resets_revision_budget_and_feeds_reasons_to_writer():
    hint = {"requested_by": "quality", "gaps": [{"criterion": "bias", "reason": "market: 단일 출처 5/8"}], "extra_rounds": 1}
    plain = normalize_state(state())
    reworked = normalize_state(state(rework_hint=hint))
    assert plain["initial_revision_rounds"] == 2 and plain["quality_feedback"] == []
    assert reworked["initial_revision_rounds"] == 0 and reworked["max_revision_rounds"] == plain["max_revision_rounds"] + 1
    assert "bias: market: 단일 출처 5/8" in section_payload("background", reworked)["quality_feedback"]
    assert "quality_feedback" not in section_payload("background", plain)
    limits = normalize_draft(draft("# 6. 한계점\n내용", "limitations"), reworked)
    assert "편중" in limits["markdown"]
