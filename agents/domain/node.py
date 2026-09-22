"""도메인 에이전트 노드: 부모 State(AppState) ↔ 도메인 내부 State 변환.

부모 그래프가 아는 것은 make_node 하나뿐이다. 내부 검색·분석 반복은 subgraph.py 에 있다.
부모 그래프에는 domain_findings, evidence_store, quality_by_perspective,
search_log_by_perspective, run_meta 만 반환한다. AppState 에 없는 키는 LangGraph가
조용히 버리므로(ISSUE.md 1-2), 그때그때 반환 키를 AppState 정의와 대조해야 한다.

AppState 는 request.errors 처럼 실행 오류를 담을 최상위 필드를 따로 두지 않는다.
그래서 도메인이 겪은 오류·페이지 사용량·프롬프트 버전은 run_meta[PERSPECTIVE] 에 담는다.
"""

from __future__ import annotations

from agents.domain.prompts import DATACENTER_REQUIREMENTS, PROMPT_VERSION
from agents.domain.subgraph import (
    PERSPECTIVE,
    DomainAgentDeps,
    DomainLocalState,
    _build_subgraph,
)

_EMPTY_SELF_CHECK = {
    "status": "failed",
    "violations": [],
    "warnings": [],
    "covered_axes": [],
    "missing_axes": [],
}

# source_type 이 이 목록에 있으면 근거를 원문 직접 증거(primary/direct)로, 아니면
# 매체를 거친 대리 증거(secondary/proxy)로 본다. quality/guard.py 가 하던 WEAK_SOURCE_TYPES
# 구분(주장 강등)을 이제 analyze 프롬프트가 대신하므로, 여기서는 Evidence 메타데이터
# 분류에만 쓴다.
STRONG_SOURCE_TYPES = {"paper", "patent", "standard"}


def project_input(state: dict) -> dict:
    """AppState 에서 이 관점이 볼 것만 추린다.

    시장·이해관계자 관점의 중간 결론을 같이 넘기면 도메인 판단이 그쪽 결론에 물든다.
    선행 단계인 기술 조사 결과와 요청 정보만 통과시킨다.
    """
    selected = state["selected_tech"]
    request = state["request"]
    return {
        "sw_name": selected["sw"]["name"],
        "hw_name": selected["hw"]["name"],
        "as_of_date": request["as_of"],
        "max_search_rounds": request["max_search_rounds"],
        "technical_summary": summarize_technical(state.get("technical_findings")),
    }


def summarize_technical(findings: dict | None) -> str:
    """선행 결과를 압축한다. 산문을 그대로 넘기면 내용이 희석된다.

    technical 에이전트가 아직 없어(ISSUE.md 4-2) PerspectiveFindings 실제 형태를
    직접 확인하지 못했다. claims/records 키가 없거나 형태가 다를 수 있어 .get 기반으로
    방어적으로 읽는다.
    """
    if not findings:
        return "(기술 조사 결과 없음 - 자체 검색 근거만으로 평가)"
    lines = [
        f"- ({c.get('technology', '?')}) {c.get('text', '')}"
        for c in findings.get("claims", [])[:12]
        if c.get("text")
    ]
    for record in findings.get("records", []):
        if "trl" in record.get("criterion", "").lower():
            value = record.get("value") or record.get("assessment", "판단보류")
            lines.append(f"- TRL({record.get('technology')}): {value} / {record.get('findings', '')}")
    return "\n".join(lines) or "(기술 조사 주장 없음)"


def _to_team_evidence(evidence: dict, claim_id: str | None) -> dict:
    """도메인 내부 Evidence(agents/domain/tools/evidence.py) → 팀 graph.state.Evidence.

    evidence_id 하나를 여러 claim 이 인용할 수 있지만 팀 스키마는 claim_id 를 단일
    필드로 둔다. 실제 연결은 Claim.evidence_ids(N:M)가 담당하므로, 여기서는 이 근거를
    처음 인용한 claim 하나만 참고용으로 남긴다.
    """
    strong = evidence["source_type"] in STRONG_SOURCE_TYPES
    return {
        "id": evidence["evidence_id"],
        "claim_id": claim_id or "",
        "doc_id": None,
        "title": evidence["title"],
        "author_or_org": evidence["author_or_organization"],
        "source_type": evidence["source_type"],
        "primary_or_secondary": "primary" if strong else "secondary",
        "direct_or_proxy": "direct" if strong else "proxy",
        "url": evidence["url"],
        "published_at": evidence.get("published_date"),
        "accessed_at": evidence["accessed_date"],
        "page_or_locator": evidence["locator"],
        "quote": evidence["quote"],
        # 도메인 관점은 지지/반박 대조를 하지 않고 판단을 뒷받침하는 근거만 모은다.
        "stance": "support",
        # 웹에서 수집한 논문·벤더 자료의 검증 단계를 세분화할 근거가 아직 없어 unknown으로 둔다.
        "evidence_level": "unknown",
        "metric_tag": None,
        "perspective": PERSPECTIVE,
        "content_hash": evidence.get("content_sha256") or "",
    }


def _to_team_claim(claim: dict) -> dict:
    technologies = claim["technology_ids"]
    technology = technologies[0] if len(technologies) == 1 else "both"
    return {
        "claim_id": claim["claim_id"],
        "technology": technology,
        "perspective": PERSPECTIVE,
        "text": claim["text"],
        "evidence_ids": claim["evidence_ids"],
        "conditions": claim["conditions"],
        "limitations": claim["limitations"],
    }


def _to_team_record(record: dict) -> dict:
    return {
        "technology": record["technology_id"],
        "perspective": PERSPECTIVE,
        "criterion": record["criterion"],
        "basis": record["basis"],
        "evidence_level": record["evidence_level"],
        "scope": record["scope"],
        "stance_counts": {},
        "evidence_ids": record["evidence_ids"],
        "assessment": record["assessment"],
        "assessment_vocab": record["assessment_vocab"],
        "value": record["value"],
        "findings": record["findings"],
        "limitations": record["limitations"],
    }


def _to_team_gaps(missing_axes: list[str], code_gaps: list[str]) -> list[dict]:
    """축 단위로 특정할 수 있는 공백과, 코드가 잡은 구조적 공백을 Gap 형태로 옮긴다."""
    # 충분성 판정과 self_check가 같은 축을 모두 보고할 수 있다.
    # 이 경우 동일 Gap을 한 번만 남기고 criterion을 축 이름으로 보존한다.
    axis_names = {axis.split(":", 1)[0].strip(): axis for axis in DATACENTER_REQUIREMENTS}

    def axis_name(value: str) -> str | None:
        if value in axis_names:
            return value
        return next((name for name, full in axis_names.items() if value == full), None)

    axes = list(dict.fromkeys(
        name for value in [*missing_axes, *code_gaps] if (name := axis_name(value)) is not None
    ))
    other_gaps = list(dict.fromkeys(g for g in code_gaps if axis_name(g) is None))
    gaps = [
        {
            "technology": "both",
            "perspective": PERSPECTIVE,
            "criterion": axis,
            "reason": "근거 부족으로 판단하지 못함",
            "missing_evidence": [],
        }
        for axis in axes
    ]
    gaps.extend(
        {
            "technology": "both",
            "perspective": PERSPECTIVE,
            "criterion": "-",
            "reason": reason,
            "missing_evidence": [],
        }
        for reason in other_gaps
    )
    return gaps


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
            "records": [],
            "self_check": {},
            "search_rounds_used": 0,
            "evidence_sufficient": False,
            "gaps": [],
            "errors": [],
        }
        try:
            result = subgraph.invoke(local)
        except Exception as exc:
            return _output(
                claims=[], records=[], evidence_store={}, search_log=[], pages_used=0,
                self_check=_EMPTY_SELF_CHECK, search_rounds=0, gaps=[],
                errors=[f"도메인 평가 서브그래프 실패: {type(exc).__name__}: {exc}"],
                status="failed",
            )

        claims = result["claims"]
        records = result["records"]
        self_check = result["self_check"] or _EMPTY_SELF_CHECK
        gaps = result["gaps"]

        if not claims and not records:
            status = "failed"
        elif gaps or self_check.get("status") != "passed":
            status = "partial"
        else:
            status = "complete"

        return _output(
            claims=claims, records=records, evidence_store=result["evidence_store"],
            search_log=result["search_log"], pages_used=result["pages_used"],
            self_check=self_check, search_rounds=result["search_rounds_used"],
            gaps=gaps, errors=result["errors"], status=status,
        )

    return domain_node


def _output(*, claims, records, evidence_store, search_log, pages_used, self_check,
            search_rounds, gaps, errors, status) -> dict:
    """AppState 업데이트. 관점 전용 키만 반환해 병렬 분기에서 충돌하지 않게 한다."""
    claim_of_evidence: dict[str, str] = {}
    for claim in claims:
        for evidence_id in claim["evidence_ids"]:
            claim_of_evidence.setdefault(evidence_id, claim["claim_id"])

    findings = {
        "perspective": PERSPECTIVE,
        "status": status,
        "records": [_to_team_record(r) for r in records],
        "claims": [_to_team_claim(c) for c in claims],
        "gaps": _to_team_gaps(self_check.get("missing_axes", []), gaps),
        "limitations": [],
        "input_evidence_ids": [],
    }

    return {
        "domain_findings": findings,
        # evidence_store 는 관점 간 공유 채널이라 dict 병합 리듀서(merge_evidence_store)로 합친다.
        "evidence_store": {
            eid: _to_team_evidence(ev, claim_of_evidence.get(eid))
            for eid, ev in evidence_store.items()
        },
        "quality_by_perspective": {
            PERSPECTIVE: {
                "status": self_check.get("status", "needs_review"),
                "violations": list(self_check.get("violations", [])),
                "warnings": list(self_check.get("warnings", [])),
                "checked_claim_ids": [c["claim_id"] for c in claims],
            }
        },
        "search_log_by_perspective": {PERSPECTIVE: search_log},
        # AppState 에는 오류·수집량을 담을 전용 최상위 필드가 없어 run_meta 에 관점별로 남긴다.
        "run_meta": {
            PERSPECTIVE: {
                "errors": errors,
                "pages_used": pages_used,
                "search_rounds_used": search_rounds,
                "prompt_version": PROMPT_VERSION,
            }
        },
    }
