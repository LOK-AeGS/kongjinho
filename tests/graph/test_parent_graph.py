"""부모 그래프 테스트. API·네트워크 없이 main.py 의 조립 방식 그대로 전체 흐름을 돌린다.

실행: python -m unittest tests.graph.test_parent_graph -v   (pytest 로도 동작)
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.stubs import cited_evidence, load_fixture, replay_node  # noqa: E402
from graph.workers import as_worker  # noqa: E402
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

    def test_supervisor_routes_from_state_with_parallel_fan_out(self):
        """첫 조사 의존성 뒤 세 관점을 같은 superstep에 병렬 디스패치한다."""
        decisions = [t for t in self.trace if t["node"] == "supervisor"]
        workers = [t for t in self.trace if t["node"] != "supervisor"]
        self.assertEqual(workers[0]["node"], "technical")
        parallel_step = {t["node"] for t in workers if t["step"] == workers[1]["step"]}
        self.assertEqual(parallel_step, {"market", "stakeholder", "domain"})
        self.assertGreaterEqual(len(decisions), 2)

    def test_no_node_errors(self):
        self.assertEqual([t for t in self.trace if t["error"]], [])

    def test_on_step_receives_each_node_output(self):
        """디버깅용 콜백이 노드마다 한 번씩, 그 노드의 반환값과 함께 호출된다."""
        seen = []
        nodes, _ = build_nodes(set(), self.fixture)
        run_graph(nodes, initial_state(), lambda record, update: seen.append((record["node"], sorted(update))))
        names = [name for name, _ in seen]
        for expected in ("supervisor", "technical", "market", "stakeholder", "domain", "synthesis", "report", "quality_eval"):
            self.assertIn(expected, names)
        self.assertIn("synthesis", dict(seen)["synthesis"])

    def test_trace_records_seconds(self):
        self.assertTrue(all(t["seconds"] is not None and t["seconds"] >= 0 for t in self.trace))

    def test_each_node_writes_only_its_keys(self):
        owned = {
            "technical": {"technical_findings", "evidence_store", "node_status"},
            "market": {"market_findings", "evidence_store", "node_status"},
            "stakeholder": {"stakeholder_findings", "evidence_store", "node_status"},
            "domain": {"domain_findings", "evidence_store", "node_status"},
            "synthesis": {"synthesis", "node_status"},
            "report": {"report_sections", "references", "quality_by_perspective", "retries", "run_meta", "node_status", "artifacts", "report_version"},
            "quality_eval": {"eval_result", "quality_by_perspective", "node_status"},
            "supervisor": {"step_count", "next", "rework", "last_decision", "node_status"},
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
        self.assertGreaterEqual(self.final["report_version"], 1)
        self.assertIsNotNone(self.final["eval_result"])

    def test_offline_run_passes_quality_and_ends_after_first_report(self):
        """수치 조건 자동 부착 후 fixture 보고서는 품질 평가를 통과해 첫 보고서로 끝난다.
        불합격 시 보고서 재작성·상한 종료는 tests/graph/test_supervisor.py 가 가짜 품질 노드로 검사한다."""
        self.assertTrue(self.final["eval_result"]["passed"])
        self.assertEqual(self.final["report_version"], 1)
        self.assertIn("품질 평가 통과", self.final["last_decision"]["reason"])

    def test_deliberate_fixture_error_is_excluded_and_disclosed(self):
        """fixture 에 일부러 넣은 '35.7%' 주장은 인용 근거 원문에 수치·조건이 없어 보고서에서 빠지고,
        빠졌다는 사실이 한계점에 공개된다(근거 없는 조건을 덧붙이지 않는다)."""
        report = self.final["quality_by_perspective"]["report"]
        self.assertFalse(any("35.7" in v for v in report["violations"]))
        path = (self.final.get("artifacts") or {}).get("report_with_ids_md")
        markdown = Path(path).read_text(encoding="utf-8") if path else \
            self.final["report_sections"].get("final_markdown_with_ids", "")
        self.assertNotIn("ITME는 처리량을 35.7% 높였다", markdown)
        self.assertIn("보고서에 싣지 않았다", markdown)

    def test_modes_describe_every_node(self):
        self.assertEqual(set(self.modes), {"technical", "market", "stakeholder", "domain", "synthesis", "report", "quality_eval"})


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
        nodes["market"] = as_worker("market", market_node(MarketAgentDeps(
            llm=NoLLM(), web_search=lambda q: [{"title": q, "url": "https://news.example.com/a", "content": q,
                                                "organization": "example", "published_date": "2026-07", "source_type": "news"}])))
        nodes["stakeholder"] = as_worker("stakeholder", stakeholder_node(client=empty_client))

        final, trace = run_graph(nodes, initial_state(fixture))
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
        nodes["domain"] = as_worker("domain", domain_node(DomainAgentDeps(llm=NoLLM(), search_provider=NoSearch())))
        final, trace = run_graph(nodes, initial_state(fixture))
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
            self.assertEqual(Path(final["artifacts"]["report_md"]).resolve(), (Path(tmp) / "report.md").resolve())
            self.assertNotIn("final_markdown", final["report_sections"])


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
        nodes["technical"] = as_worker("technical", technical_node(deps))
        final, trace = run_graph(nodes, initial_state())
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
