"""부모 그래프 테스트. API·네트워크 없이 main.py 의 조립 방식 그대로 전체 흐름을 돌린다.

실행: python -m unittest tests.graph.test_parent_graph -v   (pytest 로도 동작)
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.state import create_initial_state  # noqa: E402
from graph.stubs import cited_evidence, load_fixture, replay_node  # noqa: E402
from main import build_nodes, node_status, run_graph, steps_view  # noqa: E402


def initial_state(fixture):
    return create_initial_state(
        request={"as_of": fixture["request"]["as_of"], "language": "ko", "scope": "datacenter_inference", "max_search_rounds": 1},
        selected_tech=fixture["selected_tech"],
        corpus_manifest=[],
    )


class OfflineGraphTest(unittest.TestCase):
    """main.py 기본값(전부 오프라인)으로 부모 그래프 전체를 돌린다."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = load_fixture()
        nodes, cls.modes = build_nodes(set(), cls.fixture)
        cls.final, cls.trace = run_graph(nodes, initial_state(cls.fixture))

    def test_follows_designed_order_with_parallel_fan_out(self):
        """설계서 §8.1: ① → ②③④ 병렬(같은 step) → ⑤ → ⑥."""
        self.assertEqual(steps_view(self.trace),
                         ["technical", "market + stakeholder + domain", "synthesis", "report"])

    def test_no_node_errors(self):
        self.assertEqual([t for t in self.trace if t["error"]], [])

    def test_each_node_writes_only_its_keys(self):
        owned = {
            "technical": {"technical_findings", "evidence_store"},
            "market": {"market_findings", "evidence_store"},
            "stakeholder": {"stakeholder_findings", "evidence_store"},
            "domain": {"domain_findings", "evidence_store"},
            "synthesis": {"synthesis"},
            "report": {"report_sections", "references", "quality_by_perspective", "retries", "run_meta"},
        }
        for t in self.trace:
            self.assertTrue(set(t["updated"]) <= owned[t["node"]], t)

    def test_parallel_evidence_is_merged_without_loss(self):
        """②③④ 가 동시에 쓴 evidence_store 가 reducer 로 빠짐없이 합쳐진다."""
        expected = set()
        for p in ("technical", "market", "stakeholder", "domain"):
            expected |= cited_evidence(self.fixture[f"{p}_findings"]) & set(self.fixture["evidence_store"])
        self.assertEqual(set(self.final["evidence_store"]), expected)

    def test_synthesis_and_report_are_produced(self):
        status = node_status(self.final)
        self.assertIn(status["synthesis"], ("complete", "partial"))
        self.assertGreater(len(self.final["synthesis"]["matrix"]), 0)
        self.assertGreater(len(self.final["report_sections"]), 0)
        self.assertIn(status["report"], ("passed", "needs_review"))

    def test_report_catches_deliberate_fixture_error(self):
        """fixture 에 일부러 넣은 '35.7%' 조건 누락이 보고서 검사에서 잡힌다 (에이전트 간 검사 연결 확인)."""
        report = self.final["quality_by_perspective"]["report"]
        self.assertTrue(any("35.7" in v for v in report["violations"]))

    def test_modes_describe_every_node(self):
        self.assertEqual(set(self.modes), {"technical", "market", "stakeholder", "domain", "synthesis", "report"})


class RealNodeWiringTest(unittest.TestCase):
    """실제 에이전트 노드를 가짜 LLM·검색으로 부모 그래프에 꽂아도 끝까지 도는지."""

    def test_real_market_and_stakeholder_nodes_run_inside_graph(self):
        from agents.market import MarketAgentDeps, make_node as market_node
        from agents.stakeholder_eval import StakeholderOpinionBatch, make_node as stakeholder_node

        class NoLLM:
            def with_structured_output(self, schema):
                raise RuntimeError("오프라인: LLM 호출 없음")

        empty_client = SimpleNamespace(responses=SimpleNamespace(
            parse=lambda **_: SimpleNamespace(status="completed", output_parsed=StakeholderOpinionBatch(opinions=[]))))

        fixture = load_fixture()
        nodes, _ = build_nodes(set(), fixture)
        nodes["market"] = market_node(MarketAgentDeps(
            llm=NoLLM(), web_search=lambda q: [{"title": q, "url": "https://news.example.com/a", "content": q,
                                                "organization": "example", "published_date": "2026-07", "source_type": "news"}]))
        nodes["stakeholder"] = stakeholder_node(client=empty_client)

        final, trace = run_graph(nodes, initial_state(fixture))
        self.assertEqual(steps_view(trace), ["technical", "market + stakeholder + domain", "synthesis", "report"])
        self.assertEqual([t for t in trace if t["error"]], [])
        self.assertIsNotNone(final["market_findings"])
        self.assertIsNotNone(final["stakeholder_findings"])
        # 실제 노드가 빈 결과를 내도 평가 종합·보고서는 멈추지 않고 한계로 기록한다
        self.assertIsNotNone(final["synthesis"])
        self.assertGreater(len(final["report_sections"]), 0)


class ReplayNodeTest(unittest.TestCase):
    def test_replay_returns_only_cited_evidence(self):
        fixture = load_fixture()
        out = replay_node("market", fixture)({})
        self.assertEqual(set(out), {"market_findings", "evidence_store"})
        self.assertEqual(set(out["evidence_store"]), cited_evidence(fixture["market_findings"]) & set(fixture["evidence_store"]))


if __name__ == "__main__":
    unittest.main()
