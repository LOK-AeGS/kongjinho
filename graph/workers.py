"""기존 에이전트 노드를 supervisor worker 계약으로 감싼다."""

from __future__ import annotations

import copy
from pathlib import Path

from graph.supervisor import utc_now


PERSPECTIVE_WORKERS = ("technical", "market", "stakeholder", "domain")


def _cited_count(findings: dict | None, evidence_store: dict) -> int:
    """records·claims가 인용한 실재 근거 수(supervisor 충분도 판정과 같은 기준)."""
    findings = findings or {}
    return len({
        evidence_id
        for item in (findings.get("records") or []) + (findings.get("claims") or [])
        for evidence_id in item.get("evidence_ids", [])
        if evidence_id in evidence_store
    })


def as_worker(name: str, node_fn, *, artifact_dir: Path | None = None):
    def worker(state: dict) -> dict:
        attempts = int(((state.get("node_status") or {}).get(name) or {}).get("attempts", 0)) + 1
        view = copy.deepcopy(state)
        directive = (state.get("rework") or {}).get(name)
        if directive:
            request = dict(view.get("request") or {})
            rounds = int(directive.get("max_search_rounds", request.get("max_search_rounds", 1)))
            request["max_search_rounds"] = min(2, rounds) if name == "technical" else rounds
            view["request"] = request
            if name == "report":
                view["report_feedback"] = list(directive.get("feedback") or [])
            if name == "stakeholder":
                focus = list(directive.get("focus") or [])
                gaps = list((view.get("stakeholder_findings") or {}).get("gaps") or [])
                matched = [
                    gap for gap in gaps
                    if "/".join(
                        str(gap.get(key) or "") for key in ("technology", "criterion")
                    ).strip("/") in focus
                ]
                view["rework_hint"] = {
                    "focus_queries": focus,
                    "gaps": matched or gaps,
                }
        try:
            output = dict(node_fn(view) or {})
        except Exception as exc:
            return {"node_status": {name: {
                "status": "failed",
                "attempts": attempts,
                "completed_step": int(state.get("step_count", 0)),
                "last_error": repr(exc)[:300],
                "sufficiency": None,
                "updated_at": utc_now(),
            }}}

        if name == "report":
            sections = dict(output.get("report_sections") or {})
            if artifact_dir is not None:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                artifacts = {}
                for key, filename, artifact_key in (
                    ("final_markdown", "report.md", "report_md"),
                    ("final_markdown_with_ids", "report_with_ids.md", "report_with_ids_md"),
                ):
                    markdown = sections.pop(key, None)
                    if markdown is not None:
                        path = (artifact_dir / filename).resolve()
                        path.write_text(markdown, encoding="utf-8")
                        artifacts[artifact_key] = str(path)
                pdf_path = ((output.get("run_meta") or {}).get("report") or {}).get("pdf_path")
                if pdf_path:
                    artifacts["report_pdf"] = str(Path(pdf_path).resolve())
                output["report_sections"] = sections
                output["artifacts"] = artifacts
            output["report_version"] = int(state.get("report_version", 0)) + 1

        note = None
        key = f"{name}_findings"
        previous = state.get(key)
        if name in PERSPECTIVE_WORKERS and directive and previous and key in output:
            store = {**(state.get("evidence_store") or {}), **(output.get("evidence_store") or {})}
            before, after = _cited_count(previous, store), _cited_count(output[key], store)
            if after < before:
                # live 4차: 재작업이 결과를 악화시켰다(stakeholder 근거 1→0, market 6→3).
                # 더 나쁜 재작업 결과로 덮어쓰지 않고 이전 결과를 유지한다. 새 근거는 reducer로 합쳐 둔다.
                output[key] = previous
                note = f"재작업 결과 근거 {after}건 < 이전 {before}건 → 이전 결과 유지"

        output["node_status"] = {name: {
            "status": "done",
            "attempts": attempts,
            "completed_step": int(state.get("step_count", 0)),
            "last_error": None,
            "sufficiency": None,
            "updated_at": utc_now(),
            **({"note": note} if note else {}),
        }}
        return output

    worker.__name__ = f"worker_{name}"
    return worker
