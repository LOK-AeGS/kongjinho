"""시장성 Rubric 판정. LLM이 아니라 코드가 결정적으로 계산한다(재현성).

같은 근거를 넣으면 항상 같은 판정이 나와야 종합 에이전트가 비교 매트릭스를 신뢰할 수 있다.
"""

from __future__ import annotations

import re

from agents.market.state import Claim, InternalEvidence

CRITERIA = {"size": "시장 규모·성장성", "adoption": "상용화·채택 현황", "ecosystem": "생태계 지지"}
TECH_IDS = ("sw", "hw")
CONFIDENCE = {"충분": "high", "부분": "medium", "부족": "low"}


def compute_verdicts(claims: list[Claim], evidence: list[InternalEvidence]) -> list[dict]:
    """기술 × 세부 기준마다 판정 기록을 만든다. 직접 근거(direct_evidence) 주장만 센다."""
    ev_by_id = {e["id"]: e for e in evidence}
    out = []
    for tid in TECH_IDS:
        for key, label in CRITERIA.items():
            cs = [c for c in claims if c["basis"] == "direct_evidence" and c["technology_id"] == tid and c["aspect"] == key]
            evs = [ev_by_id[i] for c in cs for i in c["evidence_ids"] if i in ev_by_id]
            sources = {e.get("doc_id") or e["url"] for e in evs}
            non_community = [e for e in evs if e["source_type"] != "community"]

            if key == "size":
                strong = len(sources) >= 2 and any(re.search(r"\d", c["statement"]) for c in cs)
            elif key == "adoption":
                strong = False
                for c in cs:
                    e = ev_by_id.get(c["evidence_ids"][0]) if c["evidence_ids"] else None
                    if e is None or e["source_type"] == "paper":
                        continue  # 논문·시뮬레이션은 실운용으로 인정하지 않는다.
                    if c["evidence_level"] == "production" or (c["evidence_level"] == "announcement" and e["source_type"] in ("official_web", "news")):
                        strong = True
            else:
                strong = len({e["url"] for e in evs if e["source_type"] == "official_web"}) >= 2

            limits = ["공개 정보 기반 추정"]
            if not cs:
                verdict = "부족"
            elif strong and non_community:
                verdict = "충분"
            else:
                verdict = "부분"
                if strong and not non_community:
                    limits.append("블로그·커뮤니티 근거만 있어 '충분'으로 올리지 않음")
            if any(c["scope"] == "class" for c in cs):
                limits.append("기술군 수준 근거 포함")
            if any(c["evidence_level"] == "forecast" for c in cs):
                limits.append("전망치 포함, 실측 아님")
            stances = {s: sum(1 for c in cs if c["stance"] == s) for s in ("support", "counter", "neutral")}
            out.append({
                "technology_id": tid, "criterion": label, "verdict": verdict,
                "evidence_ids": sorted({i for c in cs for i in c["evidence_ids"]}),
                "scope": "direct" if any(c["scope"] == "direct" for c in cs) else ("class" if cs else "-"),
                "stance_counts": stances, "limitations": limits, "confidence": CONFIDENCE[verdict],
            })
    return out


def counter_missing(claims: list[Claim]) -> list[str]:
    return [tid for tid in TECH_IDS if not any(c["technology_id"] == tid and c["stance"] == "counter" for c in claims)]


def missing_slots(claims: list[Claim], evidence: list[InternalEvidence]) -> list[str]:
    key_of = {v: k for k, v in CRITERIA.items()}
    slots = [f"{v['technology_id']}:{key_of[v['criterion']]}" for v in compute_verdicts(claims, evidence) if v["verdict"] == "부족"]
    return slots + [f"{tid}:counter" for tid in counter_missing(claims)]


def gap_messages(claims: list[Claim], evidence: list[InternalEvidence], names: dict[str, str]) -> list[str]:
    gaps = [f"근거 부족: {names[v['technology_id']]} / {v['criterion']}" for v in compute_verdicts(claims, evidence) if v["verdict"] == "부족"]
    gaps += [f"반대·한계 근거 없음: {names[tid]}" for tid in counter_missing(claims)]
    return gaps


def imbalance_note(claims: list[Claim], names: dict[str, str]) -> str | None:
    n = {tid: sum(1 for c in claims if c["basis"] == "direct_evidence" and c["technology_id"] == tid) for tid in TECH_IDS}
    lo, hi = min(n.values()), max(n.values())
    if hi >= 3 and lo * 3 < hi:
        return f"근거 수 불균형: {names['sw']} {n['sw']}건 / {names['hw']} {n['hw']}건"
    return None
