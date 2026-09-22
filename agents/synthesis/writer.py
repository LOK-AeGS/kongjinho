"""③ 서술 작성기. 두 가지 구현이 같은 인터페이스를 가진다.

  write(payload)  -> {"claims": [DraftClaim dict], "explanations": {finding_id: (text, resolved)}}
  revise(claim, violations, payload) -> DraftClaim dict | None

TemplateWriter: LLM 없이 매트릭스·관계를 정해진 문장으로 옮긴다. 오프라인 테스트와 fixture 실행용.
OpenAIWriter  : OpenAI Responses API 구조화 출력. 실제 서술용.
"""

from __future__ import annotations

import json
import os

from agents.synthesis.prompts import PROMPT_VERSION, REVISE_SYSTEM, REVISE_USER, WRITE_SYSTEM, WRITE_USER

PERSPECTIVE_LABEL = {"technical": "기술 조사", "market": "시장", "stakeholder": "이해관계자", "domain": "도메인"}
CONFLICT_LABEL = {
    "SX1": "증거 수준 상충", "SX2": "범위 불일치", "SX3": "조건 누락", "SX4": "수치 차이",
    "SX5": "시점 차이", "SX6": "같은 근거의 반대 stance", "SX7": "TRL 입력 불일치",
}


def build_payload(state: dict) -> dict:
    """LLM 입력. 근거는 인용 quote 와 메타데이터만 넣는다."""
    store = state.get("evidence_store") or {}
    records, cited = [], set()
    for perspective, result in (state.get("findings") or {}).items():
        for r in (result or {}).get("records") or []:
            ids = [e for e in r.get("evidence_ids", []) if e in store]
            cited |= set(ids)
            records.append({"ref": f"{r['perspective']}/{r['technology']}/{r['criterion']}", "technology": r["technology"],
                            "perspective": r["perspective"], "basis": r["basis"], "scope": r["scope"],
                            "assessment": r["assessment"], "value": r.get("value"),
                            "findings": r.get("findings", ""), "evidence_ids": ids})
    findings = [{k: f[k] for k in ("id", "kind", "rule_id", "technology", "record_refs", "evidence_ids")} | {"note": f.get("_note")}
                for f in state.get("cross_findings") or []]
    for f in findings:
        cited |= set(f["evidence_ids"])
    evidence = {eid: {"quote": store[eid].get("quote", ""), "title": store[eid].get("title", ""),
                      "published_at": store[eid].get("published_at"), "evidence_level": store[eid].get("evidence_level"),
                      "direct_or_proxy": store[eid].get("direct_or_proxy"), "stance": store[eid].get("stance")}
                for eid in sorted(cited)}
    return {"as_of": state.get("as_of"), "tech_names": state.get("tech_names") or {},
            "matrix": state.get("matrix") or [], "cross_findings": findings, "records": records,
            "evidence": evidence, "gaps": state.get("gaps") or []}


class TemplateWriter:
    name = "template"
    model = None

    def write(self, payload: dict) -> dict:
        names = payload["tech_names"]
        label = lambda t: names.get(t, t) if t != "both" else f"{names.get('sw', 'SW')}·{names.get('hw', 'HW')}"
        evidence = payload["evidence"]
        claims = []

        def add(tech, text, ids):
            ids = [e for e in ids if e in evidence]
            if not ids:
                return
            tags = []
            if any(evidence[e].get("direct_or_proxy") == "proxy" for e in ids):
                tags.append("기술군 수준 근거 포함")
            if any(evidence[e].get("evidence_level") == "forecast" for e in ids):
                tags.append("전망 근거 포함")
            suffix = f" ({', '.join(tags)})" if tags else ""
            claims.append({"technology": tech, "text": f"[{label(tech)}] {text}{suffix}", "evidence_ids": ids,
                           "conditions": [], "limitations": []})

        # 같은 종류·규칙·기술의 관계는 한 문장으로 묶는다 (문장 수를 줄이고 240자 안에 담기 위해)
        groups: dict[tuple, dict] = {}
        for f in payload["cross_findings"]:
            g = groups.setdefault((f["kind"], f["rule_id"], f["technology"]), {"refs": [], "ids": []})
            g["refs"] += [r for r in f["record_refs"] if r not in g["refs"]]
            g["ids"] += [e for e in f["evidence_ids"] if e not in g["ids"]]
        for (kind, rule, tech), g in groups.items():
            refs = ", ".join(g["refs"][:2]) + (" 외" if len(g["refs"]) > 2 else "")
            if kind == "conflict":
                add(tech, f"{CONFLICT_LABEL.get(rule, '상충')} 항목이 탐지됐다: {refs}.", g["ids"])
            elif kind == "shared_evidence":
                add(tech, f"같은 근거를 공유하는 관점이 있어 독립된 확인으로 세지 않는다: {refs}.", g["ids"])
            elif kind == "complement":
                add(tech, f"서로 다른 질문에 같은 방향으로 답한 관점을 보완 관계로 기록한다: {refs}.", g["ids"])
            elif kind == "agreement":
                add(tech, f"같은 질문에 같은 방향으로 판단한 관점이 있다: {refs}.", g["ids"])

        for r in payload["records"]:
            if r["basis"] == "unknown":
                add(r["technology"], f"{PERSPECTIVE_LABEL[r['perspective']]} 관점 '{r['ref'].split('/', 2)[2]}'은 판단 보류(basis=unknown)다.",
                    r["evidence_ids"])
            elif r["perspective"] == "technical" and r.get("value"):
                add(r["technology"], f"기술 조사 관점의 TRL 추정은 {r['value']} 범위이며 공개 정보 기반 추정이다.", r["evidence_ids"])

        # 템플릿은 상충의 원인을 찾을 수 없으므로 설명을 쓰지 않는다 (resolution=unresolved 유지)
        return {"claims": claims, "explanations": {}}

    def revise(self, claim: dict, violations: list[str], payload: dict) -> dict | None:
        return None  # 템플릿 문장은 규칙대로 만들어지므로 고칠 수 없으면 제거한다


class OpenAIWriter:
    name = "openai"

    def __init__(self, model: str | None = None, client=None, max_output_tokens: int = 16000):
        from openai import OpenAI  # 오프라인 실행에서는 불러오지 않는다

        self.model = model or os.getenv("SYNTHESIS_MODEL", "gpt-4.1")
        self.client = client or OpenAI(timeout=180, max_retries=1)
        self.max_output_tokens = max_output_tokens

    def _parse(self, system: str, user: str, schema):
        response = self.client.responses.parse(model=self.model, instructions=system, input=user,
                                               text_format=schema, max_output_tokens=self.max_output_tokens, store=False)
        if response.status != "completed" or response.output_parsed is None:
            detail = getattr(response, "incomplete_details", None)
            raise RuntimeError(f"synthesis LLM 응답 미완료: status={response.status}, detail={detail}")
        return response.output_parsed

    def write(self, payload: dict) -> dict:
        from agents.synthesis.schemas import Draft

        dump = lambda x: json.dumps(x, ensure_ascii=False, indent=1)
        user = WRITE_USER.format(tech_names=dump(payload["tech_names"]), as_of=payload["as_of"],
                                 matrix=dump(payload["matrix"]), cross_findings=dump(payload["cross_findings"]),
                                 records=dump(payload["records"]), evidence=dump(payload["evidence"]), gaps=dump(payload["gaps"]))
        draft = self._parse(WRITE_SYSTEM, user, Draft)
        return {"claims": [c.model_dump() for c in draft.claims],
                "explanations": {e.finding_id: (e.explanation, e.resolved) for e in draft.explanations}}

    def revise(self, claim: dict, violations: list[str], payload: dict) -> dict | None:
        from agents.synthesis.schemas import DraftClaim

        evidence = {e: payload["evidence"][e] for e in claim.get("evidence_ids", []) if e in payload["evidence"]}
        user = REVISE_USER.format(claim=json.dumps(claim, ensure_ascii=False), violations="\n".join(violations),
                                  evidence=json.dumps(evidence or payload["evidence"], ensure_ascii=False, indent=1))
        return self._parse(REVISE_SYSTEM, user, DraftClaim).model_dump()


def writer_info(writer) -> dict:
    return {"writer": getattr(writer, "name", type(writer).__name__), "model": getattr(writer, "model", None),
            "prompt_version": PROMPT_VERSION}
