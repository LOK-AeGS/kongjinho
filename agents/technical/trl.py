"""원문 결합 관측만 사용하는 결정적 TRL Gate와 수치 검증."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TRL_DEFINITIONS = {
    1: "기본 원리 관찰",
    2: "구체적 기술 개념과 응용 정의",
    3: "핵심 기능 proof-of-concept",
    4: "구성요소의 laboratory validation",
    5: "구성요소의 relevant-environment validation",
    6: "통합 system prototype의 relevant-environment demonstration",
    7: "operational environment의 integrated pilot",
    8: "완전 통합 시스템 qualification",
    9: "지속적인 production operation",
}

ARTIFACT_RANK = {
    "unknown": 0, "principle": 1, "concept": 2, "mechanism": 3,
    "component": 4, "system_prototype": 5, "final_system": 6,
}


@dataclass(frozen=True)
class TRLResult:
    technology: str
    level: int | None
    level_range: list[int]
    stage_name: str
    verified_environment: str
    estimated_from_public_info: bool
    supporting_evidence: list[str]
    blocking_evidence: list[str]
    missing_evidence: list[str]
    gate_trace: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _rank_at_least(value: str, threshold: str) -> bool:
    return ARTIFACT_RANK.get(value, 0) >= ARTIFACT_RANK[threshold]


def _direct(observation: dict) -> bool:
    return observation.get("source_scope") == "direct" and bool(observation.get("evidence_ids"))


def _conditions(level: int, observation: dict) -> tuple[list[bool], list[str]]:
    artifact = observation.get("artifact_level", "unknown")
    environment = observation.get("environment", "unknown")
    activity = observation.get("activity", "unknown")
    evidence_level = observation.get("evidence_level", "unknown")
    direct = _direct(observation)
    if level == 1:
        return [direct, _rank_at_least(artifact, "principle")], ["직접 원리 근거", "기본 원리 관찰"]
    if level == 2:
        return [direct, _rank_at_least(artifact, "concept")], ["직접 설계 근거", "구체적 응용 개념"]
    if level == 3:
        return [direct, _rank_at_least(artifact, "mechanism"), activity in {"experiment", "pilot", "qualification", "deployment", "sustained_operation"}], ["직접 PoC 근거", "작동하는 핵심 메커니즘", "실험 결과"]
    if level == 4:
        return [direct, _rank_at_least(artifact, "component"), environment in {"lab", "relevant", "production_equivalent", "operational"}], ["직접 구성요소 근거", "구성요소 프로토타입", "실험실 이상 환경"]
    if level == 5:
        return [direct, _rank_at_least(artifact, "component"), environment in {"relevant", "production_equivalent", "operational"}, bool(observation.get("real_llm")), bool(observation.get("real_accelerator")), bool(observation.get("representative_workload"))], ["직접 relevant-environment 근거", "구성요소 프로토타입", "relevant environment", "실제 LLM", "실제 가속기", "대표 workload"]
    if level == 6:
        return [direct, _rank_at_least(artifact, "system_prototype"), environment in {"relevant", "production_equivalent", "operational"}, bool(observation.get("representative_scale"))], ["직접 system prototype 근거", "통합 system prototype", "relevant environment", "대표 규모"]
    if level == 7:
        return [direct, _rank_at_least(artifact, "system_prototype"), environment in {"production_equivalent", "operational"}, activity in {"pilot", "deployment", "sustained_operation"}, evidence_level in {"pilot", "production"}], ["직접 운영 근거", "통합 system prototype", "production-equivalent 또는 operational environment", "pilot/deployment 활동", "pilot 이상 근거 수준"]
    if level == 8:
        return [direct, artifact == "final_system", environment == "operational", activity in {"qualification", "deployment", "sustained_operation"}, bool(observation.get("operational_requirements_validated")), evidence_level == "production"], ["직접 qualification 근거", "최종형 시스템", "operational environment", "qualification/deployment 활동", "운영 요구사항 검증", "production 수준 근거"]
    return [direct, artifact == "final_system", environment == "operational", activity == "sustained_operation", evidence_level == "production", bool(observation.get("operational_requirements_validated"))], ["직접 production 근거", "최종형 시스템", "operational environment", "지속 운용", "production 수준 근거", "운영 요구사항 검증"]


def determine_trl(technology: str, observations: list[dict]) -> TRLResult:
    """각 Gate를 평가한 뒤 1부터 연속 충족된 최고 단계를 채택한다."""
    scoped = [item for item in observations if item.get("technology") == technology]
    trace: list[dict] = []
    for level in range(1, 10):
        best: dict | None = None
        best_score = -1
        best_missing: list[str] = []
        for item in scoped:
            checks, labels = _conditions(level, item)
            score = sum(checks)
            if score > best_score:
                best, best_score = item, score
                best_missing = [label for check, label in zip(checks, labels) if not check]
        satisfied = bool(best and not best_missing)
        trace.append({
            "level": level,
            "definition": TRL_DEFINITIONS[level],
            "satisfied": satisfied,
            "reason": best.get("summary", "평가 가능한 직접 관측 없음") if best else "평가 가능한 직접 관측 없음",
            "evidence_ids": list(best.get("evidence_ids", [])) if best and satisfied else [],
            "partial_evidence_ids": list(best.get("evidence_ids", [])) if best and _direct(best) else [],
            "observed_evidence_level": best.get("evidence_level", "unknown") if best else "unknown",
            "missing_evidence": best_missing or ([] if satisfied else [f"TRL {level} 직접 근거"]),
        })

    highest: int | None = None
    support: list[str] = []
    for gate in trace:
        if gate["satisfied"] and (highest is None or gate["level"] == highest + 1):
            highest = gate["level"]
            support.extend(gate["evidence_ids"])
        else:
            break

    upper = highest
    if highest is not None and highest < 9:
        next_gate = trace[highest]
        # 다음 Gate가 한 조건만 부족하면 인접 범위로 표시하되 확정 level은 올리지 않는다.
        has_meaningful_partial = bool(next_gate["partial_evidence_ids"])
        if highest + 1 >= 7:
            has_meaningful_partial = has_meaningful_partial and next_gate["observed_evidence_level"] in {"pilot", "production"}
        if len(next_gate["missing_evidence"]) == 1 and has_meaningful_partial:
            upper = highest + 1
    level_range = [] if highest is None else [highest] if upper == highest else [highest, upper]

    blocking = _unique([fact for item in scoped for fact in item.get("blocking_facts", [])])
    missing = _unique(
        [fact for item in scoped for fact in item.get("missing_facts", [])]
        + [value for gate in trace[(highest or 0):] for value in gate["missing_evidence"]]
    )
    environment = max(
        (item.get("environment", "unknown") for item in scoped if _direct(item)),
        key=lambda value: {"unknown": 0, "literature": 1, "lab": 2, "relevant": 3, "production_equivalent": 4, "operational": 5}.get(value, 0),
        default="unknown",
    )
    return TRLResult(
        technology=technology,
        level=highest,
        level_range=level_range,
        stage_name=TRL_DEFINITIONS.get(highest, "공개 근거 부족"),
        verified_environment=environment,
        estimated_from_public_info=True,
        supporting_evidence=_unique(support),
        blocking_evidence=blocking,
        missing_evidence=missing,
        gate_trace=trace,
    )


def validate_trl_record(record: dict, evidence_store: dict[str, dict]) -> list[str]:
    issues: list[str] = []
    if record.get("estimated_from_public_info") is not True:
        issues.append("TRL은 공개 정보 기반 추정임을 표시해야 함")
    level = record.get("level")
    if level is not None and not isinstance(level, int):
        issues.append("TRL level은 정수 또는 None이어야 함")
    for evidence_id in record.get("supporting_evidence", []):
        if evidence_id not in evidence_store:
            issues.append(f"존재하지 않는 TRL 근거: {evidence_id}")
    if isinstance(level, int) and level >= 7:
        operational = [evidence_store[eid] for eid in record.get("supporting_evidence", []) if eid in evidence_store and evidence_store[eid].get("evidence_level") in {"pilot", "production"}]
        if not operational:
            issues.append("TRL 7 이상에는 pilot/production 직접 근거가 필요함")
    return _unique(issues)


def validate_metric_records(metrics: list[dict]) -> tuple[list[dict], list[str]]:
    """알려진 최대값·baseline 오귀속을 표시한다. 수치를 임의로 수정하지 않는다."""
    checked: list[dict] = []
    warnings: list[str] = []
    for metric in metrics:
        item = dict(metric)
        blob = " ".join(str(item.get(key) or "") for key in ("metric", "value", "baseline", "hardware", "model", "context")).lower()
        issues: list[str] = []
        if item.get("technology") == "sw":
            if "93.3" in blob and "deepseek 67b" not in blob and "deepseek-67b" not in blob:
                issues.append("MLA 93.3% 수치의 DeepSeek 67B 비교 기준이 누락됨")
            if "5.76" in blob and not item.get("is_maximum"):
                issues.append("MLA 5.76배 수치는 maximum generation throughput임")
            if "42.5" in blob and item.get("attributable_to_selected_technology"):
                issues.append("42.5% training cost 절감을 MLA 단독 효과로 귀속할 수 없음")
        if item.get("technology") == "hw":
            if "1.80" in blob and "nvme" not in blob:
                issues.append("ITME 1.80배 수치의 NVMe-oF baseline이 누락됨")
            if "1.81" in blob and not ("recompute" in blob and ("turn 5" in blob or "turn5" in blob)):
                issues.append("ITME 1.81배 수치의 Recompute/turn 5 TTFT 조건이 누락됨")
            if "35.7" in blob and "cpu" not in blob:
                issues.append("ITME 35.7% 수치의 CPU-offload baseline이 누락됨")
            if "3.02" in blob and not item.get("is_upper_bound"):
                issues.append("3.02배 Ideal GPU memory는 ITME 실측이 아닌 상한임")
        item["validation_issues"] = issues
        checked.append(item)
        warnings.extend(issues)
    return checked, _unique(warnings)
