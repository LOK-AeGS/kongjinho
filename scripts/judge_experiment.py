"""품질 judge 모델의 결함 탐지력·일관성·비용을 비교한다.

실행 예:
  python -m scripts.judge_experiment

가격은 2026-10 실험 계획용 추정치이며 실제 청구 가격이 아니다. 실행 전에 OpenAI의
현재 가격표와 조직별 계약을 확인해야 한다. 단위는 입력/출력 1M token당 USD다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from agents.quality.checks import PROHIBITED, citation_ids, run_checks
from agents.report.state import ReportAgentDeps
from agents.report.subgraph import run_report


CRITERIA = ("groundedness", "neutrality", "bias_control", "coverage")
EXPECTED_FAILURE = {
    "clean": None,
    "neutrality_implicit": "neutrality",
    "groundedness_fabricated": "groundedness",
    "bias_onesided": "bias_control",
    "coverage_missing": "coverage",
}
LLM_OWNED_TARGETS = {
    "neutrality_implicit": "neutrality",
    "groundedness_fabricated": "groundedness",
}
LLM_CRITERIA = ("groundedness", "neutrality")

# 추정치: input/output USD per 1M tokens. 실제 API 가격과 다를 수 있다.
PRICE_USD_PER_1M = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
}

DEFAULT_STATE = Path("outputs/graph/20260922-165957-197701/final_state.json")
DEFAULT_MODELS = "gpt-4.1-mini,gpt-4.1,gpt-4o,gpt-5-mini"
DEFAULT_OUT = Path("outputs/quality/judge_experiment_v2")


def _insert_after_heading(markdown: str, heading: str, text: str) -> str:
    pattern = re.compile(rf"^{re.escape(heading)}\s*$", re.MULTILINE)
    match = pattern.search(markdown)
    if not match:
        raise ValueError(f"보고서 heading을 찾지 못함: {heading}")
    return markdown[:match.end()] + "\n\n" + text + markdown[match.end():]


def _replace_section_body(markdown: str, heading: str, body: str) -> str:
    pattern = re.compile(
        rf"(^#{{1,2}}\s+{re.escape(heading)}\s*$)(.*?)(?=^#{{1,2}}\s+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    if not pattern.search(markdown):
        raise ValueError(f"보고서 section을 찾지 못함: {heading}")
    return pattern.sub(lambda match: match.group(1) + "\n\n" + body.strip() + "\n\n", markdown, count=1)


def _single_sw_support_evidence(state: dict, fallback: str) -> str:
    store = state.get("evidence_store") or {}
    for perspective in ("technical", "market", "stakeholder", "domain"):
        findings = state.get(f"{perspective}_findings") or {}
        for item in (findings.get("records") or []) + (findings.get("claims") or []):
            if item.get("technology") != "sw":
                continue
            for evidence_id in item.get("evidence_ids") or []:
                if evidence_id in store and store[evidence_id].get("stance") != "counter":
                    return evidence_id
    return fallback


def build_variants(base_markdown: str, state: dict) -> dict[str, str]:
    """한 기준만 의도적으로 훼손한 결정적 변형을 만든다.

    - neutrality_implicit: 코드 금지어 없이 SUMMARY와 5.1에 선택 조언을 삽입한다.
    - groundedness_fabricated: 4.4에 무인용 조작 수치와 무관한 인용이 붙은 수치를 삽입한다.
    - bias_onesided: 4.3을 단일 근거 ID의 SW 우호 문장만으로 교체한다.
    - coverage_missing: 4.2 본문을 빈 문자열로 교체한다.
    """
    ids = citation_ids(base_markdown)
    if not ids:
        raise ValueError("변형에 재사용할 기존 evidence ID가 없음")
    evidence_id = ids[0]

    summary_advice = (
        f"- 데이터센터 사업자는 HBM 증설보다 MLA 계열 모델을 먼저 도입하는 편이 합리적이다. "
        f"〔근거: {evidence_id}〕"
    )
    matrix_advice = (
        f"- 운영 조직은 CXL 계층보다 MLA 기반 구성을 먼저 검토하는 편이 타당하다. "
        f"〔근거: {evidence_id}〕"
    )
    if any(word in summary_advice + matrix_advice for word in PROHIBITED):
        raise AssertionError("neutrality_implicit 변형에 코드 금지어가 포함됨")
    neutrality = _insert_after_heading(
        base_markdown,
        "# SUMMARY",
        summary_advice,
    )
    neutrality = _insert_after_heading(
        neutrality,
        "## 5.1 비교 매트릭스",
        matrix_advice,
    )

    fabricated = _insert_after_heading(
        base_markdown,
        "## 4.4 도메인 적용",
        "\n".join((
            "- 신규 메모리 경로는 데이터센터 처리량을 87.4% 높였다.",
            f"- 동일 구성은 랙 전력을 63% 절감했다. 〔근거: {evidence_id}〕",
        )),
    )

    sw_evidence = _single_sw_support_evidence(state, evidence_id)
    one_sided = _replace_section_body(
        base_markdown,
        "4.3 이해관계자",
        "\n".join(
            f"- SW 접근은 이해관계자의 도입 부담을 낮추고 운영 유연성을 높인다. 〔근거: {sw_evidence}〕"
            for _ in range(5)
        ),
    )
    missing = _replace_section_body(base_markdown, "4.2 시장성", "")
    return {
        "clean": base_markdown,
        "neutrality_implicit": neutrality,
        "groundedness_fabricated": fabricated,
        "bias_onesided": one_sided,
        "coverage_missing": missing,
    }


def evaluate_judges(
    variants: dict[str, str],
    statuses: dict[str, str],
    evidence_store: dict,
    models: list[str],
    repeats: int,
    judge_factory,
) -> list[dict]:
    records = []
    for model in models:
        try:
            judge = judge_factory(model)
            factory_error = None
        except Exception as exc:
            judge = None
            factory_error = f"{type(exc).__name__}: {exc}"
        for variant, markdown in variants.items():
            for repeat in range(1, repeats + 1):
                started = time.perf_counter()
                try:
                    if judge is None:
                        raise RuntimeError(factory_error)
                    result = judge(markdown, statuses, evidence_store)
                    criteria = result.get("criteria") or {}
                    usage = result.get("usage") or {}
                    error = None
                except Exception as exc:
                    criteria, usage = {}, {}
                    error = f"{type(exc).__name__}: {exc}"
                records.append({
                    "model": model,
                    "variant": variant,
                    "repeat": repeat,
                    "expected_failure": EXPECTED_FAILURE[variant],
                    "criteria": criteria,
                    "latency_seconds": round(time.perf_counter() - started, 4),
                    "usage": {
                        "input_tokens": int(usage.get("input_tokens", 0) or 0),
                        "output_tokens": int(usage.get("output_tokens", 0) or 0),
                        "total_tokens": int(usage.get("total_tokens", 0) or 0),
                    },
                    "error": error,
                })
    return records


def _estimated_cost(model: str, records: list[dict]) -> float | None:
    price = PRICE_USD_PER_1M.get(model)
    if not price:
        return None
    inputs = sum(item["usage"]["input_tokens"] for item in records)
    outputs = sum(item["usage"]["output_tokens"] for item in records)
    return (inputs * price["input"] + outputs * price["output"]) / 1_000_000


def compute_metrics(records: list[dict], models: list[str], repeats: int) -> dict[str, dict]:
    metrics = {}
    for model in models:
        rows = [item for item in records if item["model"] == model]
        detections = []
        false_positives = []
        grouped: dict[tuple[str, str], list[tuple[bool, int]]] = defaultdict(list)
        for item in rows:
            expected = LLM_OWNED_TARGETS.get(item["variant"])
            for criterion in LLM_CRITERIA:
                verdict = item["criteria"].get(criterion)
                if verdict is None:
                    continue
                failed = (
                    int(verdict.get("unsupported_count", 0)) > 0
                    if criterion == "groundedness"
                    else not bool(verdict["passed"])
                )
                score = int(verdict["score"])
                grouped[(item["variant"], criterion)].append((failed, score))
                if expected == criterion:
                    detections.append(failed)
                if item["variant"] == "clean":
                    false_positives.append(failed)

        consistent, score_diffs = [], []
        for variant in EXPECTED_FAILURE:
            for criterion in LLM_CRITERIA:
                values = grouped.get((variant, criterion), [])
                consistent.append(len(values) == repeats and len({value[0] for value in values}) == 1)
                score_diffs.extend(abs(left[1] - right[1]) for left, right in combinations(values, 2))
        successful = [item for item in rows if not item["error"]]
        metrics[model] = {
            "detection_rate": sum(detections) / (2 * repeats) if repeats else 0.0,
            "false_positive_rate": sum(false_positives) / len(false_positives) if false_positives else 0.0,
            "consistency_rate": sum(consistent) / len(consistent) if consistent else 0.0,
            "mean_absolute_score_diff": sum(score_diffs) / len(score_diffs) if score_diffs else 0.0,
            "mean_latency_seconds": sum(item["latency_seconds"] for item in rows) / len(rows) if rows else 0.0,
            "input_tokens": sum(item["usage"]["input_tokens"] for item in rows),
            "output_tokens": sum(item["usage"]["output_tokens"] for item in rows),
            "total_tokens": sum(item["usage"]["total_tokens"] for item in rows),
            "estimated_cost_usd": _estimated_cost(model, rows),
            "successful_calls": len(successful),
            "errors": len(rows) - len(successful),
        }
    return metrics


def detection_matrix(
    records: list[dict],
    code_results: dict[str, dict[str, dict]],
    models: list[str],
    repeats: int,
) -> list[dict]:
    matrix = []
    for model in models:
        for variant, expected in EXPECTED_FAILURE.items():
            code = code_results[variant]
            if expected is None:
                code_correct = all(value["passed"] for value in code.values())
            else:
                code_correct = not code[expected]["passed"]
            matching = [item for item in records if item["model"] == model and item["variant"] == variant]
            llm_correct, hybrid_correct = 0, 0
            llm_owned = expected in LLM_CRITERIA if expected is not None else True
            for item in matching:
                if expected is None:
                    grounded = item["criteria"].get("groundedness") or {}
                    neutral = item["criteria"].get("neutrality") or {}
                    llm_ok = bool(item["criteria"]) and int(grounded.get("unsupported_count", 1)) == 0 and bool(neutral.get("passed"))
                    hybrid_ok = llm_ok and all(value["passed"] for value in code.values())
                else:
                    verdict = item["criteria"].get(expected) or {}
                    if expected == "groundedness":
                        llm_ok = int(verdict.get("unsupported_count", 0)) > 0
                    elif expected == "neutrality":
                        llm_ok = bool(verdict) and not verdict.get("passed", True)
                    else:
                        llm_ok = False
                    hybrid_ok = (not code[expected]["passed"]) or llm_ok
                llm_correct += int(llm_ok)
                hybrid_correct += int(hybrid_ok)
            matrix.append({
                "model": model,
                "variant": variant,
                "expected": expected or "no failure",
                "code": code_correct,
                "llm_rate": (llm_correct / repeats if repeats else 0.0) if llm_owned else None,
                "hybrid_rate": hybrid_correct / repeats if repeats else 0.0,
            })
    return matrix


def _summary_table(metrics: dict[str, dict]) -> str:
    lines = [
        "| model | detection | false positive | consistency | mean score diff | latency(s) | tokens | est. USD | errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, value in metrics.items():
        cost = "n/a" if value["estimated_cost_usd"] is None else f"{value['estimated_cost_usd']:.4f}"
        lines.append(
            f"| {model} | {value['detection_rate']:.1%} | {value['false_positive_rate']:.1%} | "
            f"{value['consistency_rate']:.1%} | {value['mean_absolute_score_diff']:.2f} | "
            f"{value['mean_latency_seconds']:.2f} | {value['total_tokens']} | {cost} | {value['errors']} |"
        )
    return "\n".join(lines)


def render_markdown(metrics: dict[str, dict], matrix: list[dict], repeats: int) -> str:
    lines = [
        "# Judge model experiment",
        "",
        f"각 모델·변형을 {repeats}회 평가했다. 가격은 코드의 1M token당 추정치이며 실제 청구액이 아니다.",
        "",
        "## Per-model summary",
        "",
        _summary_table(metrics),
        "",
        "## Per-variant detection matrix",
        "",
        "clean 행은 오탐 없이 통과한 비율, 결함 행은 목표 기준을 탐지한 비율이다.",
        "",
        "| model | variant | expected | code correct | LLM correct | hybrid correct |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in matrix:
        lines.append(
            f"| {item['model']} | {item['variant']} | {item['expected']} | "
            f"{'yes' if item['code'] else 'no'} | "
            f"{'n/a' if item['llm_rate'] is None else format(item['llm_rate'], '.1%')} | {item['hybrid_rate']:.1%} |"
        )
    return "\n".join(lines) + "\n"


def _load_env(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def run_experiment(state: dict, models: list[str], repeats: int, judge_factory) -> dict:
    report_result = run_report(state, ReportAgentDeps(generation_mode="deterministic"))
    base = report_result["report_sections"]["final_markdown_with_ids"]
    variants = build_variants(base, state)
    findings = {name: state.get(f"{name}_findings") for name in ("technical", "market", "stakeholder", "domain")}
    # 변형 탐지 실험은 주입한 결함만 비교한다. production quality node는 별도로 report validator 위반을 결합한다.
    code_results = {
        name: run_checks(markdown, state.get("evidence_store") or {}, findings)
        for name, markdown in variants.items()
    }
    statuses = {name: (value or {}).get("status", "missing") for name, value in findings.items()}
    records = evaluate_judges(
        variants,
        statuses,
        state.get("evidence_store") or {},
        models,
        repeats,
        judge_factory,
    )
    metrics = compute_metrics(records, models, repeats)
    matrix = detection_matrix(records, code_results, models, repeats)
    return {
        "config": {"models": models, "repeats": repeats},
        "prices_usd_per_1m_tokens_estimate": PRICE_USD_PER_1M,
        "expected_failure": EXPECTED_FAILURE,
        "code_results": code_results,
        "judge_records": records,
        "metrics": metrics,
        "detection_matrix": matrix,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="보고서 품질 judge 모델 비교 실험")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--models", default=DEFAULT_MODELS, help="쉼표로 구분한 OpenAI 모델")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats는 1 이상이어야 합니다")
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    if not models:
        parser.error("--models에 모델을 하나 이상 지정해야 합니다")

    _load_env()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    from agents.quality.judge import make_judge

    result = run_experiment(state, models, args.repeats, make_judge)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = render_markdown(result["metrics"], result["detection_matrix"], args.repeats)
    (args.out / "results.md").write_text(markdown, encoding="utf-8")
    print(_summary_table(result["metrics"]))
    print(f"results: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
