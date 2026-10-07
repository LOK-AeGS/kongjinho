"""보고서 분량 예산·페이지 가드 테스트. API 키·네트워크 없이 실행된다."""

import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.report import subgraph  # noqa: E402
from agents.report.budget import SECTION_BUDGETS, balanced_pick, body_length  # noqa: E402
from agents.report.references import (  # noqa: E402
    collect_references,
    number_citations,
    reference_numbers,
)
from agents.report.state import BODY_SECTION_ORDER, ReportAgentDeps  # noqa: E402
from agents.report.subgraph import (  # noqa: E402
    DeterministicSectionWriter,
    _writer_context,
    normalize_state,
    run_report,
    section_payload,
)
from test_report_agent import fixture, run_offline  # noqa: E402


def inflated(copies=8):
    """관점마다 claim을 복제해 예산을 넘는 입력을 만든다."""
    state = fixture()
    for key in ("technical_findings", "market_findings", "stakeholder_findings", "domain_findings"):
        originals = state[key]["claims"]
        state[key]["claims"] = [
            {**deepcopy(claim), "claim_id": f"{claim['claim_id']}:dup{index}"}
            for index in range(copies)
            for claim in originals
        ]
    return state


class BalancedPickTests(unittest.TestCase):
    def test_keeps_both_technologies_and_a_counter(self):
        items = [{"technology": "sw", "n": i} for i in range(6)]
        items += [{"technology": "hw", "n": 10}, {"technology": "hw", "n": 11, "counter": True}]
        picked = balanced_pick(items, 3, is_counter=lambda item: item.get("counter", False))
        self.assertIn("hw", {item["technology"] for item in picked})
        self.assertIn("sw", {item["technology"] for item in picked})
        self.assertTrue(any(item.get("counter") for item in picked))
        self.assertEqual(len(picked), 3)

    def test_short_list_is_untouched(self):
        items = [{"technology": "sw"}, {"technology": "hw"}]
        self.assertEqual(balanced_pick(items, 5), items)


class SectionBudgetTests(unittest.TestCase):
    def test_deterministic_sections_respect_budget(self):
        context = normalize_state(inflated())
        writer = DeterministicSectionWriter()
        for section_id in BODY_SECTION_ORDER:
            draft = writer.write(section_id, _writer_context(section_id, context, include_normalized=True))
            budget = SECTION_BUDGETS[section_id]
            # 항목 1개만 남겨도 넘치는 경우를 빼면 상한을 지킨다.
            if draft["markdown"].count("\n- ") > 1:
                self.assertLessEqual(body_length(draft["markdown"]), budget.max_chars, section_id)

    def test_omitted_items_are_disclosed(self):
        context = normalize_state(inflated())
        draft = DeterministicSectionWriter().write(
            "stakeholder", _writer_context("stakeholder", context, include_normalized=True)
        )
        self.assertIn("분량 제한으로 전체", draft["markdown"])

    def test_payload_is_trimmed_and_carries_budget(self):
        context = normalize_state(inflated())
        payload = section_payload("market", context)
        self.assertEqual(payload["budget"]["max_items"], SECTION_BUDGETS["market"].max_items)
        self.assertLessEqual(len(payload["findings"]["claims"]), SECTION_BUDGETS["market"].max_items)
        self.assertGreater(payload["omitted"]["claims"]["total"], payload["omitted"]["claims"]["kept"])

    def test_limitations_keep_upstream_status_under_tight_budget(self):
        report = run_offline(fixture("partial_upstream"))["report"]
        limitations = next(s["markdown"] for s in report["sections"] if s["title"] == "6. 한계점")
        self.assertIn("stakeholder", limitations)
        self.assertIn("partial", limitations)


class CitationNumberingTests(unittest.TestCase):
    def setUp(self):
        self.store = {
            "a1": {"url": "https://arxiv.org/abs/2405.04434v5", "title": "DeepSeek-V2", "source_type": "paper"},
            "a2": {"url": "https://arxiv.org/html/2405.04434v2", "title": "DeepSeek-V2", "source_type": "paper"},
            "b": {"url": "https://example.com/post?utm_source=openai", "title": "Post", "source_type": "news"},
        }

    def test_arxiv_versions_share_one_reference(self):
        lines, _ = collect_references(["a1", "a2", "b"], self.store)
        self.assertEqual(len(lines), 2)
        self.assertNotIn("utm_source", lines[1])

    def test_numbers_follow_reference_order(self):
        numbers = reference_numbers(["b", "a1", "a2"], self.store)
        # 논문이 기타보다 먼저 정렬되므로 논문이 1번이다.
        self.assertEqual(numbers, {"a1": 1, "a2": 1, "b": 2})
        lines, _ = collect_references(["b", "a1", "a2"], self.store)
        markdown = "# A\n\n- 주장 〔근거: a2, b〕\n\n# REFERENCE\n\n" + "\n".join(f"- {line}" for line in lines)
        display = number_citations(markdown, numbers)
        self.assertIn("- 주장 [1, 2]", display)
        self.assertIn("- [1] 논문", display)
        self.assertIn("- [2] 기타", display)
        self.assertNotIn("〔근거", display)


class PageGuardTests(unittest.TestCase):
    def test_report_records_page_count(self):
        result = run_offline(fixture())
        report = result["report"]
        self.assertLessEqual(report["page_count"], report["page_limit"])
        self.assertEqual(report["page_count_method"], "estimate")
        self.assertEqual(result["report_sections"]["final_markdown"], report["display_markdown"])
        self.assertIn("〔근거", result["report_sections"]["final_markdown_with_ids"])

    def test_guard_shrinks_then_stops_with_issue(self):
        with mock.patch.object(subgraph, "PAGE_LIMIT", 1):
            result = run_report(inflated(), ReportAgentDeps(generation_mode="deterministic"))
        report = result["report"]
        guard = report["page_guard"]
        self.assertEqual(len(guard), subgraph.MAX_PAGE_GUARD_ROUNDS + 1)
        self.assertLessEqual(guard[-1]["pages"], guard[0]["pages"])
        self.assertTrue(any(item["code"] == "page_limit_exceeded" for item in result["issues"]))
        self.assertEqual(report["quality_status"], "needs_review")


if __name__ == "__main__":
    unittest.main()
