"""기존 에이전트 노드를 supervisor worker 계약으로 감싼다."""

from __future__ import annotations

import copy
from pathlib import Path

from graph.supervisor import utc_now


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

        output["node_status"] = {name: {
            "status": "done",
            "attempts": attempts,
            "completed_step": int(state.get("step_count", 0)),
            "last_error": None,
            "sufficiency": None,
            "updated_at": utc_now(),
        }}
        return output

    worker.__name__ = f"worker_{name}"
    return worker
