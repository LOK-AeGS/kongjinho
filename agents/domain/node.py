"""도메인 에이전트 노드: 부모 State ↔ 도메인 내부 State 변환.

부모 그래프가 아는 것은 make_node 하나뿐이다. 내부 검색·분석 반복은 subgraph.py 에 있다.
부모 그래프에는 domain_findings 와 도메인 전용 부속 키만 반환한다.
"""

from __future__ import annotations

from agents.domain.prompts import PROMPT_VERSION
from agents.domain.subgraph import (
    PERSPECTIVE,
    DomainAgentDeps,
    DomainLocalState,
    _apply_quality,
    _build_subgraph,
)


def project_input(state: dict) -> dict:
    """부모 State에서 이 관점이 볼 것만 추린다.

    시장·이해관계자 관점의 중간 결론을 같이 넘기면 도메인 판단이 그쪽 결론에 물든다.
    선행 단계인 기술 조사 결과와 요청 정보만 통과시킨다.
    """
    request = state["request"]
    return {
        "sw_name": request["sw"]["name"],
        "hw_name": request["hw"]["name"],
        "as_of_date": request["as_of_date"],
        "max_search_rounds": request["max_search_rounds"],
        "technical_summary": summarize_technical(state.get("technical_findings")),
    }


def summarize_technical(findings: dict | None) -> str:
    """선행 결과를 압축한다. 산문을 그대로 넘기면 내용이 희석된다."""
    if not findings:
        return "(기술 조사 결과 없음 - 자체 검색 근거만으로 평가)"
    lines = [
        f"- ({'/'.join(c['technology_ids'])}) {c['topic']}: {c['statement']}"
        for c in findings.get("claims", [])[:12]
    ]
    for trl in findings.get("trl_estimates", []):
        level = trl["level"] if trl["level"] is not None else "판단보류"
        lines.append(f"- TRL({trl['technology_id']}): {level} / {trl['caveat']}")
    return "\n".join(lines) or "(기술 조사 주장 없음)"




def make_node(deps: DomainAgentDeps):
    """graph.build.build_graph(domain=...) 에 주입할 노드 함수를 만든다."""
    subgraph = _build_subgraph(deps)

    def domain_node(state: dict) -> dict:
        projected = project_input(state)
        local: DomainLocalState = {
            **projected,
            "questions": [],
            "evidence_store": {},
            "source_texts": {},
            "search_log": [],
            "pages_used": 0,
            "condition_notes": [],
            "claims": [],
            "fits": [],
            "search_rounds_used": 0,
            "evidence_sufficient": False,
            "gaps": [],
            "errors": [],
        }
        try:
            result = subgraph.invoke(local)
        except Exception as exc:
            return _output(
                claims=[], fits=[], evidence_store={}, search_log=[], pages_used=0,
                guard=None, lint=None, judge=None, search_rounds=0, gaps=[],
                errors=[f"도메인 평가 서브그래프 실패: {type(exc).__name__}: {exc}"],
                status="failed",
            )

        claims, fits, guard, lint, judge, gaps = _apply_quality(result, deps)
        if not claims:
            status = "failed"
        elif gaps or not guard.passed or not judge.passed:
            status = "partial"
        else:
            status = "complete"

        return _output(
            claims=claims, fits=fits, evidence_store=result["evidence_store"],
            search_log=result["search_log"], pages_used=result["pages_used"],
            guard=guard, lint=lint, judge=judge,
            search_rounds=result["search_rounds_used"], gaps=gaps, errors=result["errors"],
            status=status,
        )

    return domain_node


def _output(*, claims, fits, evidence_store, search_log, pages_used, guard, lint, judge,
            search_rounds, gaps, errors, status) -> dict:
    """부모 State 업데이트. 관점별 키로 분리해 병렬 분기에서 충돌하지 않게 한다."""
    return {
        "domain_findings": {
            "claims": claims,
            "fits": fits,
            "cited_evidence_ids": sorted({eid for c in claims for eid in c["evidence_ids"]}),
            "completion": {
                "status": status,
                "search_rounds_used": search_rounds,
                "revision_rounds_used": 0,
                "pages_used": pages_used,
                "gaps": gaps,
                "errors": errors,
            },
        },
        # evidence_store 는 관점 간 공유 채널이라 dict 병합 리듀서로 합친다.
        "evidence_store": evidence_store,
        "quality_by_perspective": {
            PERSPECTIVE: {
                "guard": guard.to_dict() if guard else None,
                "lint": lint.to_dict() if lint else None,
                "judge": judge.to_dict() if judge else None,
                "prompt_version": PROMPT_VERSION,
            }
        },
        "search_log_by_perspective": {PERSPECTIVE: search_log},
    }
