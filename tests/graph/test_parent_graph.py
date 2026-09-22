"""부모 그래프 테스트. API·네트워크 없이 main.py 의 조립 방식 그대로 전체 흐름을 돌린다.

실행: python -m unittest tests.graph.test_parent_graph -v   (pytest 로도 동작)
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.stubs import cited_evidence, load_fixture, replay_node  # noqa: E402
from main import build_nodes, initial_state as main_initial_state, node_status, run_graph, steps_view  # noqa: E402


def initial_state(fixture=None):
    """main.py 와 같은 초기값: 팀 고정 입력 + Pool A manifest."""
    return main_initial_state(rounds=1)


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

    def test_on_step_receives_each_node_output(self):
        """디버깅용 콜백이 노드마다 한 번씩, 그 노드의 반환값과 함께 호출된다."""
        seen = []
        nodes, _ = build_nodes(set(), self.fixture)
        run_graph(nodes, initial_state(), lambda record, update: seen.append((record["node"], sorted(update))))
        self.assertEqual(sorted(n for n, _ in seen), sorted(["technical", "market", "stakeholder", "domain", "synthesis", "report"]))
        self.assertIn("synthesis", dict(seen)["synthesis"])

    def test_trace_records_seconds(self):
        self.assertTrue(all(t["seconds"] is not None and t["seconds"] >= 0 for t in self.trace))

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


    def test_real_domain_node_failure_does_not_stop_graph(self):
        """실제 도메인 노드가 LLM 없이 실패해도 failed 결과로 끝나고, 그래프는 종합·보고서까지 간다."""
        from agents.domain import DomainAgentDeps, make_node as domain_node

        class NoLLM:
            def with_structured_output(self, schema):
                raise RuntimeError("오프라인: LLM 호출 없음")

            def invoke(self, *args, **kwargs):
                raise RuntimeError("오프라인: LLM 호출 없음")

        class NoSearch:
            def search(self, query):
                raise RuntimeError("오프라인: 검색 없음")

        fixture = load_fixture()
        nodes, _ = build_nodes(set(), fixture)
        nodes["domain"] = domain_node(DomainAgentDeps(llm=NoLLM(), search_provider=NoSearch()))
        final, trace = run_graph(nodes, initial_state(fixture))
        self.assertEqual(steps_view(trace), ["technical", "market + stakeholder + domain", "synthesis", "report"])
        self.assertEqual([t for t in trace if t["error"]], [])
        self.assertEqual(final["domain_findings"]["status"], "failed")
        self.assertIn("domain", final["quality_by_perspective"])
        self.assertIsNotNone(final["synthesis"])


class PdfOutputTest(unittest.TestCase):
    def test_report_node_writes_pdf(self):
        try:
            import reportlab  # noqa: F401
        except ImportError:
            self.skipTest("reportlab 미설치")
        import tempfile
        fixture = load_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "report.pdf"
            nodes, _ = build_nodes(set(), fixture, pdf)
            final, _ = run_graph(nodes, initial_state(fixture))
            self.assertTrue(pdf.is_file() and pdf.stat().st_size > 0)
            self.assertEqual(Path(final["run_meta"]["report"]["pdf_path"]).resolve(), pdf.resolve())
            self.assertIn("final_markdown", final["report_sections"])


class TechnicalNodeTest(unittest.TestCase):
    def test_initial_state_uses_team_fixed_input(self):
        from agents.technical.config import DEFAULT_SELECTED_TECH
        state = initial_state()
        self.assertEqual(state["selected_tech"], DEFAULT_SELECTED_TECH)
        self.assertGreater(len(state["corpus_manifest"]), 0)  # Pool A 고정 코퍼스

    def test_real_technical_node_runs_inside_graph(self):
        """실제 기술 조사 노드를 가짜 retriever·Tavily·analyzer 로 꽂아도 1단계에서 끝까지 돈다."""
        import importlib
        fakes = importlib.import_module("tests.agents.technical.test_technical_agent")
        from agents.technical import make_node as technical_node

        deps, _ = fakes._deps()
        fixture = load_fixture()
        nodes, _ = build_nodes(set(), fixture)
        nodes["technical"] = technical_node(deps)
        final, trace = run_graph(nodes, initial_state())
        self.assertEqual(steps_view(trace), ["technical", "market + stakeholder + domain", "synthesis", "report"])
        self.assertEqual([t for t in trace if t["error"]], [])
        self.assertNotEqual(final["technical_findings"]["status"], "failed", final["technical_findings"]["limitations"])
        self.assertTrue(any(r["criterion"] == "trl" for r in final["technical_findings"]["records"]))
        self.assertIsNotNone(final["synthesis"])


class ReplayNodeTest(unittest.TestCase):
    def test_replay_returns_only_cited_evidence(self):
        fixture = load_fixture()
        out = replay_node("market", fixture)({})
        self.assertEqual(set(out), {"market_findings", "evidence_store"})
        self.assertEqual(set(out["evidence_store"]), cited_evidence(fixture["market_findings"]) & set(fixture["evidence_store"]))


if __name__ == "__main__":
    unittest.main()
