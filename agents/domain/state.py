"""도메인 관점이 부모 그래프에 요구하는 State 확장.

팀 `state.py` 를 직접 고치지 않고 별도로 둔다. 다른 다섯 에이전트가 아직 작업 중이라
공유 스키마를 지금 바꾸면 그쪽 코드가 깨지기 때문이다. 에이전트를 합칠 때 이 정의를
`state.py` 로 옮기면 된다.

현재 팀 설계와 다른 점:
- evidence 를 각 findings 안의 list 로 두지 않고 최상위 dict 로 뺀다.
  리스트에 operator.add 로 쌓으면 재시도할 때 같은 근거가 중복으로 들어가고,
  관점끼리 같은 출처를 찾았을 때 같은 자료가 두 벌 남는다.
  evidence_id 가 출처 신원에서 결정적으로 유도되므로 dict 병합이 멱등해진다.
- 품질 결과를 quality_by_perspective 로 분리한다. 관점마다 검사 통과 여부가 다르고,
  종합 단계가 "어느 관점 결과를 얼마나 믿을지" 판단하려면 관점별로 남아야 한다.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from agents.domain.rag.evidence import merge_evidence

Perspective = Literal["technical", "market", "stakeholder", "domain"]


def merge_by_perspective(left: dict, right: dict) -> dict:
    """관점 키가 서로 달라 충돌하지 않는다. 같은 관점이 재실행되면 최신 값으로 대체한다."""
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class DomainCompletion(TypedDict):
    status: Literal["complete", "partial", "failed"]
    search_rounds_used: int
    revision_rounds_used: int
    pages_used: int  # 웹 수집 누적 페이지. 200 한도 확인용
    gaps: list[str]
    errors: list[str]


class DomainFit(TypedDict):
    technology_id: Literal["sw", "hw"]
    domain: str
    requirements: list[str]
    assessment: Literal["suitable", "conditional", "unsuitable", "unknown"]
    claim_ids: list[str]
    limitations: list[str]


class DomainFindings(TypedDict):
    claims: list[dict]
    fits: list[DomainFit]
    cited_evidence_ids: list[str]  # 근거 본문은 evidence_store 에 있고 여기선 ID만 참조
    completion: DomainCompletion


class DomainAwareState(TypedDict, total=False):
    """부모 PipelineState 에 더해져야 하는 키.

    evidence_store 와 quality_by_perspective 는 여러 관점이 동시에 쓰므로 리듀서가 필요하다.
    domain_findings 는 작성자가 하나라 리듀서 없이 마지막 쓰기가 유효하다.
    """

    evidence_store: Annotated[dict[str, dict], merge_evidence]
    quality_by_perspective: Annotated[dict[str, dict], merge_by_perspective]
    search_log_by_perspective: Annotated[dict[str, list], merge_by_perspective]
    domain_findings: DomainFindings


def to_team_findings(findings: DomainFindings, evidence_store: dict[str, dict]) -> dict:
    """팀 `state.py` 의 DomainFindings 형태로 되돌린다(에이전트 합칠 때 사용).

    팀 스키마는 evidence 를 findings 안에 list 로 담으므로, 인용한 근거만 추려 넣는다.
    """
    cited = set(findings["cited_evidence_ids"])
    completion = dict(findings["completion"])
    completion.pop("pages_used", None)  # 팀 AgentCompletion 에는 없는 필드
    return {
        "claims": findings["claims"],
        "evidence": [evidence_store[eid] for eid in sorted(cited) if eid in evidence_store],
        "completion": completion,
        "fits": findings["fits"],
    }
