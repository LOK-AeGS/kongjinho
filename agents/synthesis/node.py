"""평가 종합 노드: 부모 AppState ↔ 내부 State 변환.

부모 그래프가 아는 것은 make_node 하나뿐이다. 내부 4단계 흐름은 subgraph.py 에 있다.
  읽는 키: request, selected_tech, technical/market/stakeholder/domain_findings, evidence_store
  쓰는 키: synthesis
"""

from __future__ import annotations

from datetime import date

from agents.synthesis.rules import PERSPECTIVES, RULES_VERSION
from agents.synthesis.subgraph import build_subgraph
from agents.synthesis.writer import TemplateWriter, writer_info


def project_input(state: dict) -> dict:
    """부모 State 에서 종합에 필요한 것만 추린다. 새 검색·새 사실은 만들지 않으므로 입력이 전부다."""
    request = state.get("request") or {}
    selected = state.get("selected_tech") or {}
    return {
        "as_of": request.get("as_of") or date.today().isoformat(),
        "tech_names": {t: (selected.get(t) or {}).get("short_name") or (selected.get(t) or {}).get("name") or t
                       for t in ("sw", "hw")},
        "findings": {p: state.get(f"{p}_findings") for p in PERSPECTIVES},
        "evidence_store": state.get("evidence_store") or {},
    }


def to_result(local: dict, writer) -> dict:
    """내부 State → AppState.synthesis (SynthesisResult). 내부 전용 필드(_note 등)는 뺀다."""
    findings = [{k: v for k, v in f.items() if not k.startswith("_")} for f in local.get("cross_findings") or []]
    limitations = list(local.get("limitations") or [])
    limitations += [f"오류: {e}" for e in local.get("errors") or []]
    if local.get("dropped_sentences"):
        limitations.append(f"중립성 검사로 제거된 문장 {len(local['dropped_sentences'])}개 (dropped_sentences 참고)")
    status = local.get("status", "failed")
    if status == "complete" and (local.get("errors") or (local.get("quality") or {}).get("status") != "passed"):
        status = "partial"
    return {
        "status": status,
        "as_of": local.get("as_of", ""),
        "input_hash": local.get("input_hash", ""),
        "matrix": local.get("matrix") or [],
        "cross_findings": findings,
        "contrast_table": local.get("contrast_table") or [],
        "summary_claims": local.get("summary_claims") or [],
        "gaps": local.get("gaps") or [],
        "imbalance": local.get("imbalance") or {},
        "retry_requests": local.get("retry_requests") or [],
        "limitations": limitations,
        "dropped_sentences": local.get("dropped_sentences") or [],
        "meta": writer_info(writer) | {
            "rules_version": RULES_VERSION,
            "quality": local.get("quality") or {},
            "counts": {
                "matrix_cells": len(local.get("matrix") or []),
                "conflicts": sum(f["kind"] == "conflict" for f in findings),
                "shared_evidence": sum(f["kind"] == "shared_evidence" for f in findings),
                "agreements": sum(f["kind"] == "agreement" for f in findings),
                "complements": sum(f["kind"] == "complement" for f in findings),
            },
        },
    }


def _step_summary(node: str, update: dict) -> str:
    """노드가 만든 것을 한 줄로. 실행 경로 확인용."""
    if node == "matrix":
        return (f"status={update.get('status')}, 매트릭스 {len(update.get('matrix') or [])}칸, "
                f"공백 {len(update.get('gaps') or [])}, 재실행 요청 {len(update.get('retry_requests') or [])}")
    if node == "relations":
        kinds = {}
        for f in update.get("cross_findings") or []:
            kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
        return f"관계 {sum(kinds.values())}건 {kinds}, 대조표 {len(update.get('contrast_table') or [])}행"
    if node == "write":
        errors = update.get("errors") or []
        return f"초안 문장 {len(update.get('summary_claims') or [])}개" + (f", 오류 {errors}" if errors else "")
    if node == "review":
        q = update.get("quality") or {}
        return f"검사 {q.get('checked')}, 유지 {q.get('kept')}, 재생성 {q.get('revised')}, 제거 {q.get('dropped')} → {q.get('status')}"
    return ", ".join(sorted(update))


def make_node(writer=None):
    """부모 그래프에 등록할 노드 함수를 만든다.

    writer=None 이면 LLM 없이 템플릿 문장으로 서술한다 (TemplateWriter).
    실제 서술은 agents.synthesis.writer.OpenAIWriter() 를 넘긴다.
    """
    writer = writer or TemplateWriter()
    subgraph = build_subgraph(writer)

    def synthesis_node(state: dict) -> dict:
        local = project_input(state)
        result, trace = dict(local), []
        try:
            # stream 으로 돌려 노드 실행 순서와 각 단계 산출물을 meta.trace 에 남긴다 (설계대로 돌았는지 확인용)
            for mode, chunk in subgraph.stream(local, stream_mode=["updates", "values"]):
                if mode == "values":
                    result = chunk
                    continue
                for name, update in chunk.items():
                    trace.append({"step": len(trace) + 1, "node": name, "output": _step_summary(name, update or {})})
        except Exception as exc:  # 예외를 밖으로 던지지 않고 failed 결과로 반환한다 (규칙 8)
            result = {**result, "status": "failed", "errors": [f"평가 종합 실패: {type(exc).__name__}: {exc}"]}
        out = to_result(result, writer)
        out["meta"]["trace"] = trace
        return {"synthesis": out}

    return synthesis_node
