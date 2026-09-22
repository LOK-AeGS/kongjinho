"""agents/stakeholder_eval.py 단위 테스트. 실제 OpenAI 호출 없이 stub 클라이언트로 검증한다."""
import unittest
from types import SimpleNamespace

from agents.stakeholder_eval import (
    GROUPS,
    StakeholderOpinion,
    StakeholderOpinionBatch,
    build_search_plan,
    find_missing_groups,
    flag_bias,
    make_node,
    run_stakeholder_eval,
    to_perspective_findings,
)

SELECTED_TECH = {
    "sw": {"name": "DeepSeek-V2 MLA", "technology": "sw"},
    "hw": {"name": "ITME CXL", "technology": "hw"},
}


def opinion(**overrides) -> StakeholderOpinion:
    base = dict(
        speaker="Jane Operator",
        affiliation="Example Datacenter Co",
        stance="support",
        summary="배포 후 지연시간이 개선됐다고 밝혔다.",
        source_url="https://example.com/post",
        source_title="Example Post",
        published_date="2026-01-01",
        quote="Latency improved after rollout.",
        primary_or_secondary="primary",
        direct_or_proxy="direct",
        evidence_level="production",
    )
    base.update(overrides)
    return StakeholderOpinion(**base)


class StubClient:
    """(technology, group) 조합별로 미리 정해둔 응답을 돌려주는 가짜 OpenAI 클라이언트."""

    def __init__(self, responses_by_key: dict):
        self.responses_by_key = responses_by_key
        self.calls = []
        self.models = []
        self.responses = SimpleNamespace(parse=self._parse)

    def _parse(self, *, model, instructions, input, tools, tool_choice, text_format, max_output_tokens, store):
        self.calls.append(input)
        self.models.append(model)
        # 매 호출은 input JSON 문자열에 technology/group 텍스트를 담고 있다.
        for key, opinions in self.responses_by_key.items():
            technology, group_label = key
            if f'"{technology}"' in input and group_label in input:
                return SimpleNamespace(status="completed", output_parsed=StakeholderOpinionBatch(opinions=opinions))
        return SimpleNamespace(status="completed", output_parsed=StakeholderOpinionBatch(opinions=[]))


class BuildSearchPlanTests(unittest.TestCase):
    def test_plan_covers_every_technology_and_group(self):
        plan = build_search_plan(SELECTED_TECH, "datacenter")
        combos = {(item["technology"], item["group"]) for item in plan}
        expected = {(tech, group) for tech in ("sw", "hw") for group in GROUPS}
        self.assertEqual(combos, expected)


class FindMissingGroupsTests(unittest.TestCase):
    def test_reports_only_combos_with_zero_opinions(self):
        plan = build_search_plan(SELECTED_TECH, "datacenter")
        found = [{"technology": "sw", "group": "operator"}]
        missing = find_missing_groups(found, plan)
        self.assertNotIn(("sw", "operator"), {(m["technology"], m["group"]) for m in missing})
        self.assertIn(("hw", "operator"), {(m["technology"], m["group"]) for m in missing})


class FlagBiasTests(unittest.TestCase):
    def test_flags_speaker_affiliated_with_the_technology(self):
        raw = [{**opinion(affiliation="DeepSeek-V2 MLA Team").model_dump(), "technology": "sw", "group": "supplier"}]
        flagged = flag_bias(raw, SELECTED_TECH)
        self.assertIsNotNone(flagged[0]["bias_note"])

    def test_does_not_flag_unrelated_speaker(self):
        raw = [{**opinion().model_dump(), "technology": "sw", "group": "operator"}]
        flagged = flag_bias(raw, SELECTED_TECH)
        self.assertIsNone(flagged[0]["bias_note"])


class ToPerspectiveFindingsTests(unittest.TestCase):
    def test_builds_contract_shaped_output(self):
        raw = flag_bias(
            [{**opinion().model_dump(), "technology": "sw", "group": "operator"}], SELECTED_TECH
        )
        missing = [{"technology": "hw", "group": "investor", "query": "q"}]
        findings, evidence_store = to_perspective_findings(raw, missing)

        self.assertEqual(findings["perspective"], "stakeholder")
        self.assertEqual(findings["status"], "partial")
        self.assertEqual(len(findings["claims"]), 1)
        self.assertEqual(len(findings["records"]), 1)
        self.assertEqual(len(findings["gaps"]), 1)
        self.assertEqual(set(findings["input_evidence_ids"]), set(evidence_store))

        evidence = next(iter(evidence_store.values()))
        expected_keys = {
            "id", "claim_id", "doc_id", "title", "author_or_org", "source_type",
            "primary_or_secondary", "direct_or_proxy", "url", "published_at", "accessed_at",
            "page_or_locator", "quote", "stance", "evidence_level", "metric_tag",
            "perspective", "content_hash",
        }
        self.assertEqual(set(evidence), expected_keys)


class RunStakeholderEvalTests(unittest.TestCase):
    def test_runs_end_to_end_with_stub_client_and_retries_missing_groups(self):
        client = StubClient(
            {
                ("sw", "데이터센터 운영자 / 서빙 엔지니어"): [opinion()],
                # 나머지 조합은 첫 시도에서 빈 결과 -> 재검색에서도 빈 결과 -> Gap으로 남는다.
            }
        )
        findings, evidence_store, search_log = run_stakeholder_eval(
            SELECTED_TECH, "datacenter", technical_findings=None, client=client
        )

        self.assertEqual(findings["status"], "partial")
        self.assertEqual(len(findings["claims"]), 1)
        self.assertEqual(len(evidence_store), 1)
        # 8개 질의 + 누락된 7개 조합 재검색 = 15회 호출
        self.assertEqual(len(search_log), 15)
        self.assertEqual(set(client.models), {"gpt-4.1-mini"})
        self.assertTrue(any(entry.get("retry") for entry in search_log))


class MakeNodeTests(unittest.TestCase):
    def test_returns_only_app_state_keys(self):
        client = StubClient({("sw", "데이터센터 운영자 / 서빙 엔지니어"): [opinion()]})
        node = make_node(client=client)
        state = {
            "selected_tech": SELECTED_TECH,
            "domain": "datacenter",
            "technical_findings": None,
        }
        result = node(state)
        self.assertEqual(set(result), {"stakeholder_findings", "evidence_store", "search_log_by_perspective"})
        self.assertIn("stakeholder", result["search_log_by_perspective"])


if __name__ == "__main__":
    unittest.main()
