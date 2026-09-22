"""평가 종합 서브그래프: 분기·루프 없는 4단계 직선 흐름.

  START → matrix → relations → write → review → END
          (코드)    (코드)      (LLM)   (코드 + 위반 문장만 LLM 1회)

설계서 §8.6 그림(10개 노드, 분기 2곳, 루프 1곳)을 줄인 이유는 docs/SYNTHESIS_AGENT.md 참고.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.synthesis.matrix import build_matrix
from agents.synthesis.relations import analyze_relations
from agents.synthesis.review import review_claims, review_context
from agents.synthesis.state import SynthesisLocalState
from agents.synthesis.writer import build_payload

CLAIM_PREFIX = "synthesis:claim:"


def _write(state: dict, writer) -> dict:
    """③ 서술. 입력 관점이 하나도 없으면(status=failed) 쓰지 않는다."""
    if state.get("status") == "failed":
        return {"summary_claims": [], "errors": []}
    try:
        draft = writer.write(build_payload(state))
    except Exception as exc:  # 서술 실패는 결과를 버리지 않고 오류로 남긴다 (규칙 8)
        return {"summary_claims": [], "errors": [f"서술 단계 실패: {type(exc).__name__}: {exc}"]}
    return {"summary_claims": draft["claims"], "errors": [], "_explanations": draft["explanations"]}


def _review(state: dict, writer) -> dict:
    """④ C1~C7 검사, 위반 문장만 1회 재생성, 그래도 위반이면 제거."""
    payload = build_payload(state)
    ctx = review_context(state)
    result = review_claims(state.get("summary_claims") or [], state.get("_explanations") or {}, ctx, writer, payload)

    findings = []
    for f in state.get("cross_findings") or []:
        f = dict(f)
        if f["id"] in result["explanations"]:
            text, resolved = result["explanations"][f["id"]]
            f["explanation"] = text
            f["resolution"] = "explained" if resolved else "unresolved"
        findings.append(f)

    claims = []
    for n, c in enumerate(result["claims"], 1):
        perspectives = [r["perspective"] for e in c.get("evidence_ids", [])
                        for r in ctx["records_by_evidence"].get(e, [])]
        claims.append({
            "claim_id": f"{CLAIM_PREFIX}{n:03d}",
            "technology": c.get("technology", "both"),
            # Claim.perspective 는 단일 관점만 표현할 수 있다. 인용 근거의 첫 관점을 쓴다 (docs 참고).
            "perspective": perspectives[0] if perspectives else "technical",
            "text": c["text"],
            "evidence_ids": list(c.get("evidence_ids", [])),
            "conditions": list(c.get("conditions") or []),
            "limitations": list(c.get("limitations") or []),
        })

    total = len(state.get("summary_claims") or [])
    if not claims and state.get("status") != "failed":
        quality = "failed"  # 쓸 수 있는 요약 문장이 하나도 남지 않음
    elif result["dropped"]:
        quality = "needs_review"
    else:
        quality = "passed"
    return {
        "summary_claims": claims,
        "cross_findings": findings,
        "dropped_sentences": result["dropped"],
        "quality": {"status": quality, "checked": total, "kept": len(claims), "revised": result["revised"],
                    "dropped": len(result["dropped"])},
    }


class _State(SynthesisLocalState, total=False):
    _explanations: dict  # write → review 전달용 (최종 결과에는 없음)


def build_subgraph(writer):
    graph = StateGraph(_State)
    graph.add_node("matrix", build_matrix)
    graph.add_node("relations", analyze_relations)
    graph.add_node("write", lambda s: _write(s, writer))
    graph.add_node("review", lambda s: _review(s, writer))
    graph.add_edge(START, "matrix")
    graph.add_edge("matrix", "relations")
    graph.add_edge("relations", "write")
    graph.add_edge("write", "review")
    graph.add_edge("review", END)
    return graph.compile()
