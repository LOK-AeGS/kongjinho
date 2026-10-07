"""보고서 에이전트 계약 테스트. API 키·네트워크·LangGraph 없이 실행된다."""

import json
import importlib.util
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agents.report.state import ReportAgentDeps  # noqa: E402
from agents.report.node import report_agent  # noqa: E402
from agents.report.subgraph import (  # noqa: E402
    DeterministicSectionWriter,
    _writer_context,
    normalize_state,
    run_report,
)
from agents.report.validators import extract_citations, validate_section  # noqa: E402
from agents.report.writer import LLMSectionOutput, LLMSectionWriter  # noqa: E402


FIXTURE = Path(__file__).with_name("fixtures") / "report_cases.json"


def fixture(case="complete"):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    state = deepcopy(payload["base"])
    config = payload["cases"][case]
    for key, status in config.get("status", {}).items():
        state[key]["status"] = status
    for key, gap in config.get("gap", {}).items():
        state[key]["gaps"].append({
            "technology": "both",
            "perspective": state[key]["perspective"],
            "criterion": "fixture gap",
            "reason": gap,
            "missing_evidence": [],
        })
    replacement = config.get("replace_evidence")
    if replacement:
        for result_name in ("technical_findings", "market_findings", "stakeholder_findings", "domain_findings"):
            for claim in state[result_name]["claims"]:
                if claim["claim_id"] == replacement["claim_id"]:
                    claim["evidence_ids"] = [replacement["evidence_id"]]
    return state


def run_offline(state):
    return run_report(state, ReportAgentDeps(generation_mode="deterministic"))


def node_offline(state, **kwargs):
    return report_agent(
        state,
        deps=ReportAgentDeps(generation_mode="deterministic", **kwargs),
    )


class RecordingWriter(DeterministicSectionWriter):
    def __init__(self):
        self.writes = []
        self.repairs = []

    def write(self, section_id, context):
        self.writes.append(section_id)
        draft = super().write(section_id, context)
        if section_id == "technology_overview":
            draft["markdown"] = draft["markdown"].replace("최대 35.7%", "35.7%")
        return draft

    def repair(self, section_id, draft, issues, context):
        self.repairs.append(section_id)
        return super().repair(section_id, draft, issues, context)


class HallucinatingWriter:
    """LLM이 payload 밖의 문서 ID·상태값·번호를 반환한 경우를 재현한다."""

    receives_full_context = True

    def __init__(self):
        self.base = DeterministicSectionWriter()

    def write(self, section_id, context):
        draft = self.base.write(section_id, context)
        if section_id == "limitations":
            draft["markdown"] = "# 6. 한계점\n\n- 공개 정보에 한정한다."
        draft["markdown"] += (
            "\n- 잘못된 인용. "
            "〔근거: arxiv:2606.12556v2, not_found, imbalance, 1〕"
        )
        draft["evidence_ids"] += ["1", "not_found"]
        draft["claim_ids"] += ["12"]
        return draft

    def repair(self, section_id, draft, issues, context):
        return self.write(section_id, context)


class PersistentUngroundedWriter:
    """부분 수정 뒤에도 입력에 없는 측정값을 반복하는 LLM을 재현한다."""

    receives_full_context = True

    def __init__(self):
        self.base = DeterministicSectionWriter()

    def write(self, section_id, context):
        draft = self.base.write(section_id, context)
        if section_id == "background":
            draft["markdown"] += "\n- 공개 근거에 없는 처리량 77.7% 향상."
        return draft

    def repair(self, section_id, draft, issues, context):
        return self.write(section_id, context)


class ReportAgentTests(unittest.TestCase):
    def test_required_sections(self):
        report = run_offline(fixture())["report"]
        for heading in (
            "# SUMMARY", "# 1. 분석 배경", "# 2. 기술 선정", "# 3. 기술 개요",
            "# 4. 관점별 평가", "## 4.1 기술 성숙도(TRL)", "## 4.2 시장성",
            "## 4.3 이해관계자", "## 4.4 도메인 적용", "# 5. 시사점",
            "## 5.1 비교 매트릭스", "## 5.2 적용 조건 대조표",
            "## 5.3 관점 간 상충 지점", "## 5.4 공유 근거 및 보완 관계",
            "## 5.5 남은 확인 과제", "# 6. 한계점", "# REFERENCE",
        ):
            self.assertIn(heading, report["markdown"])

    def test_summary_is_first(self):
        self.assertTrue(run_offline(fixture())["report"]["markdown"].startswith("# SUMMARY"))

    def test_reference_is_last(self):
        markdown = run_offline(fixture())["report"]["markdown"].rstrip()
        self.assertGreater(markdown.rfind("# REFERENCE"), markdown.rfind("# 6. 한계점"))
        self.assertNotIn("\n# ", markdown[markdown.rfind("# REFERENCE") + 1 :])

    def test_unknown_evidence_is_rejected(self):
        result = run_offline(fixture("invalid_evidence"))
        self.assertEqual(result["report"]["quality_status"], "needs_review")
        self.assertTrue(any(item["code"] == "unknown_evidence" for item in result["issues"]))
        self.assertNotIn("ev-missing", result["report"]["cited_evidence_ids"])

    def test_only_used_evidence_becomes_reference(self):
        report = run_offline(fixture())["report"]
        self.assertNotIn("ev-unused", report["cited_evidence_ids"])
        self.assertNotIn("Unused source", "\n".join(report["references"]))

    def test_reference_is_deduplicated(self):
        report = run_offline(fixture("duplicate_reference"))["report"]
        self.assertIn("ev-mla", report["cited_evidence_ids"])
        self.assertIn("ev-mla-adoption", report["cited_evidence_ids"])
        self.assertEqual(sum("DeepSeek-V2" in line for line in report["references"]), 1)

    def test_numeric_grounding(self):
        class UngroundedWriter(DeterministicSectionWriter):
            def write(self, section_id, context):
                draft = super().write(section_id, context)
                if section_id == "background":
                    draft["markdown"] += "\n- 확인되지 않은 77.7% 성능 수치."
                return draft

        state = fixture()
        state["retries"]["report"] = 2
        result = run_report(
            state,
            ReportAgentDeps(writer=UngroundedWriter(), writer_receives_full_context=True),
        )
        self.assertFalse(any(
            item["code"] == "numeric_grounding" and item["blocking"]
            for item in result["issues"]
        ))
        self.assertTrue(any(
            item["code"] == "llm_section_replaced_by_deterministic"
            and "numeric_grounding" in item["message"]
            for item in result["issues"]
        ))
        self.assertEqual(result["report"]["quality_status"], "passed")
        self.assertEqual(result["report"]["completion"]["revision_rounds_used"], 2)

    def test_key_metric_conditions(self):
        context = normalize_state(fixture())
        draft = {
            "section_id": "technology_overview",
            "title": "3. 기술 개요",
            "markdown": "# 3. 기술 개요\n\n- ITME는 35.7% 처리량 향상. 〔근거: ev-itme〕",
            "claim_ids": ["technical:claim:003"],
            "evidence_ids": ["ev-itme"],
        }
        issues = validate_section(draft, context)
        self.assertTrue(any(item["code"] == "metric_35_7_context" for item in issues))

    def test_prohibited_comparison_language(self):
        context = normalize_state(fixture())
        draft = {
            "section_id": "background", "title": "1. 분석 배경",
            "markdown": "# 1. 분석 배경\n\n- MLA가 승자다.",
            "claim_ids": [], "evidence_ids": [],
        }
        self.assertTrue(any(item["code"] == "prohibited_comparison" for item in validate_section(draft, context)))

    def test_partial_upstream_is_exposed(self):
        report = run_offline(fixture("partial_upstream"))["report"]
        limitations = next(section["markdown"] for section in report["sections"] if section["title"] == "6. 한계점")
        self.assertIn("stakeholder", limitations)
        self.assertIn("partial", limitations)
        self.assertEqual(report["completion"]["status"], "partial")

    def test_missing_data_is_not_invented(self):
        report = run_offline(fixture("evidence_gap"))["report"]
        self.assertIn("실운용 채택 규모 미확인", report["markdown"])
        self.assertNotIn("실운용 채택 규모가 존재하지 않는다", report["markdown"])

    def test_only_invalid_section_is_repaired(self):
        writer = RecordingWriter()
        result = run_report(
            fixture("numeric_violation"),
            ReportAgentDeps(writer=writer, writer_receives_full_context=True),
        )
        self.assertEqual(writer.repairs, ["technology_overview"])
        self.assertEqual(result["repair_log"], ["technology_overview"])
        self.assertNotEqual(writer.writes[0], "summary")
        self.assertIn("summary", writer.writes)
        self.assertEqual(result["report"]["quality_status"], "passed")

    def test_complete_fixture_passes(self):
        result = run_offline(fixture())
        blocking = [item for item in result["issues"] if item["blocking"]]
        self.assertEqual(blocking, [], blocking)
        self.assertEqual(result["report"]["quality_status"], "passed")
        self.assertEqual(
            result["finalization_steps"],
            ["summary", "citations", "references", "quality", "decision"],
        )

    def test_external_writer_context_is_section_scoped(self):
        context = _writer_context(
            "market", normalize_state(fixture()), include_normalized=False
        )
        self.assertEqual(set(context), {"payload", "prompt"})
        self.assertNotIn("evidence_store", context)
        self.assertIn("market:claim:001", str(context["payload"]))

    def test_llm_draft_unknown_ids_are_removed_and_limitations_are_completed(self):
        state = fixture("partial_upstream")
        payload = _writer_context(
            "market", normalize_state(state), include_normalized=False
        )["payload"]
        self.assertNotIn("source_ids", str(payload["technologies"]))

        result = run_report(
            state,
            ReportAgentDeps(
                writer=HallucinatingWriter(), writer_receives_full_context=True
            ),
        )
        known = set(normalize_state(state)["evidence_store"])
        self.assertLessEqual(
            set(extract_citations(result["report"]["markdown"])), known
        )
        removed = [
            item for item in result["issues"]
            if item["code"] == "removed_unknown_citation"
        ]
        self.assertTrue(removed)
        self.assertTrue(all(not item["blocking"] for item in removed))
        self.assertIn("arxiv:2606.12556v2", removed[0]["message"])
        limitations = next(
            section["markdown"] for section in result["report"]["sections"]
            if section["title"] == "6. 한계점"
        )
        self.assertIn(
            "stakeholder upstream 상태는 partial이며 관련 공백을 최종 판단에 반영해야 한다.",
            limitations,
        )
        self.assertFalse(any(
            item["code"] == "upstream_status_hidden" for item in result["issues"]
        ))

    def test_persistent_llm_numeric_violation_uses_deterministic_section(self):
        result = run_report(
            fixture(),
            ReportAgentDeps(
                writer=PersistentUngroundedWriter(),
                writer_receives_full_context=True,
            ),
        )
        background = next(
            section for section in result["report"]["sections"]
            if section["section_id"] == "background"
        )
        self.assertNotIn("77.7%", background["markdown"])
        self.assertFalse(any(
            item["blocking"] and item["section_id"] == "background"
            and item["code"] == "numeric_grounding"
            for item in result["issues"]
        ))
        replacements = [
            item for item in result["issues"]
            if item["code"] == "llm_section_replaced_by_deterministic"
        ]
        self.assertEqual(len(replacements), 1)
        self.assertEqual(replacements[0]["section_id"], "background")
        self.assertIn("numeric_grounding", replacements[0]["message"])

    def test_node_returns_only_confirmed_app_state_keys(self):
        update = node_offline(fixture())
        self.assertEqual(
            set(update),
            {"report_sections", "references", "quality_by_perspective", "retries", "run_meta"},
        )
        self.assertIn("final_markdown", update["report_sections"])
        self.assertEqual(update["quality_by_perspective"]["report"]["status"], "passed")
        self.assertEqual(update["retries"]["report"], 0)
        self.assertEqual(
            update["run_meta"]["report"]["finalization_steps"],
            ["summary", "citations", "references", "quality", "decision"],
        )
        self.assertEqual(
            update["run_meta"]["report"]["generation"]["mode"],
            "deterministic",
        )
        self.assertEqual(
            set(next(iter(update["references"].values()))),
            {"evidence_id", "title", "author_or_org", "published_at", "url", "locator"},
        )

    @unittest.skipUnless(importlib.util.find_spec("reportlab"), "reportlab 미설치")
    def test_pdf_output_is_recorded_in_run_meta(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.pdf"
            update = node_offline(fixture(), pdf_output_path=target)
            self.assertTrue(target.is_file())
            self.assertEqual(target.read_bytes()[:4], b"%PDF")
            self.assertEqual(
                update["run_meta"]["report"]["pdf_path"],
                str(target.resolve()),
            )

    def test_llm_writer_invokes_prompt_and_returns_structured_draft(self):
        class FakeStructuredModel:
            def __init__(self):
                self.messages = []

            def invoke(self, messages):
                self.messages.append(messages)
                return {
                    "markdown": "# 4.2 시장성\n\n- 시장 근거를 조건부로 기술했다. 〔근거: ev-market〕",
                    "claim_ids": ["market:claim:001"],
                    "evidence_ids": ["ev-market"],
                }

        class FakeLLM:
            def __init__(self):
                self.structured = FakeStructuredModel()

            def with_structured_output(self, schema):
                self.schema = schema
                return self.structured

        llm = FakeLLM()
        writer = LLMSectionWriter(
            llm,
            model="test-model",
            provider="test-provider",
            temperature=0,
        )
        context = _writer_context(
            "market", normalize_state(fixture()), include_normalized=False
        )
        draft = writer.write("market", context)

        self.assertEqual(draft["claim_ids"], ["market:claim:001"])
        self.assertIn("섹션 규칙", llm.structured.messages[0][1][1])
        self.assertIn("# 4.2 시장성", llm.structured.messages[0][1][1])

    def test_run_report_defaults_to_injected_llm(self):
        class FakeStructuredModel:
            def __init__(self):
                self.messages = []

            def invoke(self, messages):
                self.messages.append(messages)
                prompt = messages[1][1]
                heading = prompt.split("반드시 사용할 첫 제목: ", 1)[1].splitlines()[0]
                return {
                    "markdown": f"{heading}\n\n- 자료 미확인",
                    "claim_ids": [],
                    "evidence_ids": [],
                }

        class FakeLLM:
            def __init__(self):
                self.structured = FakeStructuredModel()

            def with_structured_output(self, schema):
                return self.structured

        llm = FakeLLM()
        result = run_report(
            fixture(),
            ReportAgentDeps(llm=llm, model="test-model", model_provider="test"),
        )

        self.assertEqual(result["generation"]["mode"], "llm")
        self.assertEqual(result["generation"]["model"], "test-model")
        self.assertEqual(len(llm.structured.messages), 14)
        self.assertTrue(
            all("섹션 규칙" in messages[1][1] for messages in llm.structured.messages)
        )


if __name__ == "__main__":
    unittest.main()
