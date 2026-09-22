"""이해관계자 평가 에이전트 (간소화 버전).

agents/stakeholder/ (이해관계자 검색 에이전트)와는 별개의 새 에이전트다.
같은 목적(이해관계자 반응 조사)을 훨씬 단순한 구조로 구현한다: LangGraph 서브그래프 대신
위에서 아래로 순서대로 실행되는 함수 5개로 구성했다. 무엇이, 왜 달라졌는지는
docs/STAKEHOLDER_EVAL_AGENT.md 에 정리했다.

전체 흐름 (docs/STAKEHOLDER_AGENT.md 흐름도의 상자 이름을 괄호로 표시):
    1) build_search_plan      - 이해관계자별 검색 계획
    2) search_group_opinions  - 웹 검색·원문 열람 + 발언 맥락 추출 + 분류 (한 번의 LLM 호출로 통합)
    3) flag_bias              - 편향 점검 (규칙 기반)
    4) find_missing_groups    - 관점이 누락되었는가? (최대 1회만 재검색)
    5) to_perspective_findings - 팀 공통 State(graph/state.py) 계약 형식으로 변환
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict

Technology = Literal["sw", "hw"]
Group = Literal["competitor", "operator", "supplier", "investor"]

GROUPS: tuple[Group, ...] = ("competitor", "operator", "supplier", "investor")

GROUP_LABELS: dict[Group, str] = {
    "competitor": "경쟁 기술 진영",
    "operator": "데이터센터 운영자 / 서빙 엔지니어",
    "supplier": "메모리·서버 공급사",
    "investor": "투자·애널리스트",
}


# =========================================================
# 구조화 출력 모델 (LLM이 검색 결과를 이 형태로 채운다)
# =========================================================

class StakeholderOpinion(BaseModel):
    """웹 검색으로 찾은 이해관계자 발언 하나.

    technology/group은 어떤 질의로 찾았는지 코드가 이미 알고 있으므로 LLM에게 다시
    묻지 않는다 (LLM이 잘못 재분류해 질의 의도와 어긋나는 것을 막기 위함).
    """

    model_config = ConfigDict(extra="forbid")

    speaker: str
    affiliation: str
    stance: Literal["support", "counter", "neutral"]
    summary: str  # 한국어 발언 요약
    source_url: str
    source_title: str
    published_date: str | None
    quote: str

    # graph/state.py의 Evidence 계약을 채우기 위한 분류. 별도 검증 로직 없이
    # LLM이 함께 판단하게 해서 계약은 지키되 코드는 단순하게 유지한다.
    primary_or_secondary: Literal["primary", "secondary"]
    direct_or_proxy: Literal["direct", "proxy"]
    evidence_level: Literal["forecast", "announcement", "pilot", "production", "unknown"]


class StakeholderOpinionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    opinions: list[StakeholderOpinion]


SEARCH_INSTRUCTIONS = """당신은 지정된 기술에 대한 특정 이해관계자 그룹의 공개 반응을 조사합니다.
웹 검색 도구로 실제 자료를 찾고, 그 자료에 실제로 있는 내용만 사용해 발언을 구조화하세요.
자료에 없는 내용을 지어내지 마세요. 해당 반응을 찾지 못하면 opinions를 빈 목록으로 두세요.
quote는 실제 출처 페이지에 있는 짧은 원문 그대로 옮기세요(한 문장 이내로 짧게).
opinions는 가장 대표적인 것 최대 3개까지만 담으세요. 응답 길이 제한에 걸리지 않도록 간결하게 답하세요."""


# =========================================================
# 1단계: 이해관계자별 검색 계획
# =========================================================

def build_search_plan(selected_tech: dict[str, dict], domain: str) -> list[dict]:
    """기술 × 그룹 조합마다 검색 질의를 하나씩 만든다."""
    plan = []
    for technology, tech_spec in selected_tech.items():
        name = tech_spec["name"] if isinstance(tech_spec, dict) else tech_spec
        for group in GROUPS:
            query = f"{name} {domain} {GROUP_LABELS[group]} 반응 평가 지지 비판 채택"
            plan.append({"technology": technology, "group": group, "query": query})
    return plan


def summarize_technical(findings: dict | None) -> str:
    """기술 조사 에이전트 결과를 짧게 요약해 이해관계자 검색의 맥락으로 넘긴다."""
    if not findings:
        return "기술 조사 결과 없음"
    lines = [claim["text"] for claim in findings.get("claims", [])[:5]]
    return " / ".join(lines) if lines else "기술 조사 결과 없음"


# =========================================================
# 2단계: 웹 검색 + 발언 추출 + 분류 (LLM 호출 1회)
# =========================================================

def _reasoning_kwargs(model: str) -> dict:
    """추론 모델이면 추론을 쓸 수 있는 가장 낮은 값으로 낮춘다.

    gpt-5 계열은 추론 토큰이 max_output_tokens 를 함께 소모해서 답변이 잘리고 검색이 실패한다
    (ISSUE 3-1a 와 같은 양상). 다만 이 호출은 web_search 를 쓰는데, API 가
    "tools cannot be used with reasoning.effort 'minimal': web_search" 로 거부하므로
    minimal 이 아니라 low 가 하한이다. (gpt-4.1 계열은 이 파라미터 자체를 거부한다.)
    """
    if model.startswith("gpt-5") and "chat" not in model:
        return {"reasoning": {"effort": "low"}}
    return {}


def search_group_opinions(
    client,
    plan_item: dict,
    technical_findings: dict | None,
    *,
    model: str = "gpt-5-mini",
) -> list[StakeholderOpinion]:
    """한 (기술, 그룹) 조합에 대해 실제 웹 검색을 수행하고 발언을 구조화해 돌려준다."""
    import json

    context = json.dumps(
        {
            "technology": plan_item["technology"],
            "group": GROUP_LABELS[plan_item["group"]],
            "query": plan_item["query"],
            "technical_findings_summary": summarize_technical(technical_findings),
        },
        ensure_ascii=False,
    )
    response = client.responses.parse(
        model=model,
        instructions=SEARCH_INSTRUCTIONS,
        input=context,
        tools=[{"type": "web_search"}],
        tool_choice="required",
        text_format=StakeholderOpinionBatch,
        max_output_tokens=8000,
        store=False,
        **_reasoning_kwargs(model),
    )
    if response.status != "completed" or response.output_parsed is None:
        return []
    return response.output_parsed.opinions


# =========================================================
# 3단계: 편향 점검 (규칙 기반)
# =========================================================

def flag_bias(opinions: list[dict], selected_tech: dict[str, dict]) -> list[dict]:
    """발언자·소속이 검색 대상 기술의 이름과 겹치면 자기 홍보 가능성을 표시한다."""
    tech_names = {
        str(spec["name"] if isinstance(spec, dict) else spec).lower()
        for spec in selected_tech.values()
    }
    flagged = []
    for opinion in opinions:
        haystack = f"{opinion['speaker']} {opinion['affiliation']}".lower()
        is_self_promo = any(name and name in haystack for name in tech_names)
        flagged.append(
            {
                **opinion,
                "bias_note": "자기 홍보 가능성 (발언자가 해당 기술 관련 조직 소속으로 보임)"
                if is_self_promo
                else None,
            }
        )
    return flagged


# =========================================================
# 4단계: 관점 누락 확인
# =========================================================

def find_missing_groups(opinions: list[dict], plan: list[dict]) -> list[dict]:
    """발언을 하나도 찾지 못한 (기술, 그룹) 조합을 돌려준다."""
    found = {(o["technology"], o["group"]) for o in opinions}
    return [item for item in plan if (item["technology"], item["group"]) not in found]


# =========================================================
# 5단계: 팀 공통 State 계약 형식으로 변환
# =========================================================

def _evidence_id(quote: str, url: str) -> str:
    return "ev:" + hashlib.sha256(f"{url}|{quote}".encode()).hexdigest()[:24]


def _claim_id(technology: str, group: str, speaker: str, quote: str) -> str:
    raw = f"{technology}|{group}|{speaker}|{quote}"
    return "claim:stakeholder:" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def to_perspective_findings(
    opinions: list[dict],
    missing: list[dict],
) -> tuple[dict, dict]:
    """graph/state.py의 PerspectiveFindings + evidence_store 형태로 변환한다."""
    accessed_at = datetime.now(timezone.utc).isoformat()

    evidence_store: dict[str, dict] = {}
    claims: list[dict] = []
    for opinion in opinions:
        evidence_id = _evidence_id(opinion["quote"], opinion["source_url"])
        claim_id = _claim_id(
            opinion["technology"], opinion["group"], opinion["speaker"], opinion["quote"]
        )

        evidence_store[evidence_id] = {
            "id": evidence_id,
            "claim_id": claim_id,
            "doc_id": None,
            "title": opinion["source_title"],
            "author_or_org": opinion["speaker"] or opinion["affiliation"] or "unknown",
            "source_type": "web",
            "primary_or_secondary": opinion["primary_or_secondary"],
            "direct_or_proxy": opinion["direct_or_proxy"],
            "url": opinion["source_url"],
            "published_at": opinion["published_date"],
            "accessed_at": accessed_at,
            "page_or_locator": "web_search_result",
            "quote": opinion["quote"],
            "stance": opinion["stance"],
            "evidence_level": opinion["evidence_level"],
            "metric_tag": None,
            "perspective": "stakeholder",
            "content_hash": evidence_id,
        }

        limitations = [opinion["bias_note"]] if opinion.get("bias_note") else []
        claims.append(
            {
                "claim_id": claim_id,
                "technology": opinion["technology"],
                "perspective": "stakeholder",
                "text": f"[{GROUP_LABELS[opinion['group']]}] {opinion['speaker']}: {opinion['summary']}",
                "evidence_ids": [evidence_id],
                "conditions": [],
                "limitations": limitations,
            }
        )

    records: list[dict] = []
    technologies = sorted({opinion["technology"] for opinion in opinions})
    for technology in technologies:
        for group in GROUPS:
            group_opinions = [
                o for o in opinions if o["technology"] == technology and o["group"] == group
            ]
            if not group_opinions:
                continue
            stance_counts = {"support": 0, "counter": 0, "neutral": 0}
            for opinion in group_opinions:
                stance_counts[opinion["stance"]] += 1
            dominant_stance = max(stance_counts, key=stance_counts.get)
            records.append(
                {
                    "technology": technology,
                    "perspective": "stakeholder",
                    "criterion": GROUP_LABELS[group],
                    "basis": "direct",
                    "evidence_level": group_opinions[0]["evidence_level"],
                    "scope": "direct",
                    "stance_counts": stance_counts,
                    "evidence_ids": [
                        _evidence_id(o["quote"], o["source_url"]) for o in group_opinions
                    ],
                    "assessment": dominant_stance,
                    "assessment_vocab": "support/counter/neutral",
                    "value": None,
                    "findings": (
                        f"{GROUP_LABELS[group]} 발언 {len(group_opinions)}건 중 "
                        f"{dominant_stance} 우세"
                    ),
                    "limitations": [],
                }
            )

    gaps = [
        {
            "technology": item["technology"],
            "perspective": "stakeholder",
            "criterion": GROUP_LABELS[item["group"]],
            "reason": "웹 검색에서 해당 조합의 발언을 찾지 못함",
            "missing_evidence": [],
        }
        for item in missing
    ]

    if not claims and not gaps:
        status = "failed"
    elif gaps:
        status = "partial"
    else:
        status = "complete"

    findings = {
        "perspective": "stakeholder",
        "status": status,
        "records": records,
        "claims": claims,
        "gaps": gaps,
        "limitations": [],
        "input_evidence_ids": list(evidence_store),
    }
    return findings, evidence_store


# =========================================================
# 진입 함수
# =========================================================

def run_stakeholder_eval(
    selected_tech: dict[str, dict],
    domain: str,
    technical_findings: dict | None = None,
    *,
    client=None,
    model: str = "gpt-4.1-mini",
) -> tuple[dict, dict, list[dict]]:
    """1~5단계를 순서대로 실행해 (findings, evidence_store, search_log)를 돌려준다."""
    if client is None:
        from openai import OpenAI

        client = OpenAI()

    plan = build_search_plan(selected_tech, domain)
    search_log: list[dict] = []
    opinions: list[dict] = []

    for item in plan:
        found = search_group_opinions(client, item, technical_findings, model=model)
        search_log.append(
            {"technology": item["technology"], "group": item["group"], "query": item["query"], "found": len(found)}
        )
        opinions.extend(
            {**o.model_dump(), "technology": item["technology"], "group": item["group"]}
            for o in found
        )

    # 관점이 누락되었는가? 예 -> 질의를 바꿔 한 번만 더 검색한다 (원래 설계는 한도까지 반복하지만
    # 여기서는 최대 1회로 고정해 로직을 단순하게 유지한다).
    missing = find_missing_groups(opinions, plan)
    for item in missing:
        retry_item = {**item, "query": item["query"] + " 원문 출처 공식 발표"}
        found = search_group_opinions(client, retry_item, technical_findings, model=model)
        search_log.append(
            {
                "technology": item["technology"],
                "group": item["group"],
                "query": retry_item["query"],
                "found": len(found),
                "retry": True,
            }
        )
        opinions.extend(
            {**o.model_dump(), "technology": item["technology"], "group": item["group"]}
            for o in found
        )
    missing = find_missing_groups(opinions, plan)

    opinions = flag_bias(opinions, selected_tech)
    findings, evidence_store = to_perspective_findings(opinions, missing)
    return findings, evidence_store, search_log


def make_node(client=None, model: str = "gpt-5-mini"):
    """부모 그래프(graph/build.py)의 stakeholder= 자리에 꽂을 노드 함수를 만든다.

    부모 State에서 이 관점이 필요로 하는 것만 추려 쓴다 (다른 관점의 중간 결론은
    보지 않는다 - agents/domain/node.py의 project_input과 동일한 원칙).
    """

    def stakeholder_eval_node(state: dict) -> dict:
        findings, evidence_store, search_log = run_stakeholder_eval(
            state["selected_tech"],
            state["domain"],
            state.get("technical_findings"),
            client=client,
            model=model,
        )
        return {
            "stakeholder_findings": findings,
            "evidence_store": evidence_store,
            "search_log_by_perspective": {"stakeholder": search_log},
        }

    return stakeholder_eval_node
