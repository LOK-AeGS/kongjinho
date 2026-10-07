"""Supervisor 패턴 테스트. 가짜 노드만 쓰므로 API·네트워크가 필요 없다.

실행: python -m pytest tests/graph/test_supervisor.py -q   (unittest 로도 동작)
확인하는 것: 라우팅이 State 에 따라 달라짐 / 근거 부족 관점에만 재작업 / 예외·부족이 계속돼도 종료 / 품질 미달 Loop /
             step 상한 / LLM 제안 guard 와 폴백 / 워커→워커 직접 edge 없음.
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from graph.build import build_graph  # noqa: E402
from graph.quality import make_quality_node  # noqa: E402
from graph.state import append_capped, create_initial_state  # noqa: E402
from graph.supervisor import guard, make_llm_proposer, make_supervisor, options, rule_choice  # noqa: E402

REQUEST = {"as_of": "2026-09-22", "language": "ko", "scope": "test", "max_search_rounds": 1}


def new_state(**kw):
    return create_initial_state(request=REQUEST, selected_tech={}, corpus_manifest=[], **kw)


def evidence(perspective, i, host=None):
    host = host or f"src{i}.example.com"
    return {"id": f"ev:{perspective}:{i}", "claim_id": f"c:{perspective}:{i}", "doc_id": None, "title": "t", "author_or_org": "a",
            "source_type": "web", "primary_or_secondary": "primary", "direct_or_proxy": "direct", "url": f"https://{host}/{perspective}/{i}",
            "published_at": "2026-01", "accessed_at": "2026-09-22", "page_or_locator": "b", "quote": "q", "stance": "support",
            "evidence_level": "unknown", "metric_tag": None, "perspective": perspective, "content_hash": "h"}


def findings(perspective, status="complete", gaps=()):
    return {"perspective": perspective, "status": status, "records": [], "claims": [], "gaps": list(gaps),
            "limitations": [], "input_evidence_ids": []}


class Harness:
    """관점별로 호출 횟수를 세고, 호출 번째마다 근거 건수·예외를 정할 수 있는 가짜 워커 모음."""

    def __init__(self, plan=None, report_text=None):
        self.plan = plan or {}  # {perspective: [n_evidence or "raise", ...]} 호출 순서대로, 모자라면 마지막 값 반복
        self.calls = {}
        self.inputs = {}
        self.report_text = report_text

    def perspective(self, p):
        def node(state):
            n_call = self.calls[p] = self.calls.get(p, 0) + 1
            self.inputs.setdefault(p, []).append(state)
            seq = self.plan.get(p, [3])
            n = seq[min(n_call, len(seq)) - 1]
            if n == "raise":
                raise RuntimeError("boom")
            return {f"{p}_findings": findings(p), "evidence_store": {e["id"]: e for e in (evidence(p, i) for i in range(n))}}
        return node

    def synthesis(self, state):
        self.calls["synthesis"] = self.calls.get("synthesis", 0) + 1
        return {"synthesis": {"status": "complete", "retry_requests": [], "meta": {}}}

    def report(self, state):
        self.calls["report"] = self.calls.get("report", 0) + 1
        store = state["evidence_store"]
        ids = sorted({next(i for i, e in store.items() if e["perspective"] == p) for p in ("technical", "market", "stakeholder", "domain") if any(e["perspective"] == p for e in store.values())})
        text = self.report_text(self.calls["report"]) if self.report_text else "# SUMMARY\n각 관점의 평가는 다음과 같다. 〔근거: " + ", ".join(ids) + "〕\n# REFERENCE\n"
        return {"report_sections": {"final_markdown": text}, "references": {},
                "quality_by_perspective": {"report": {"status": "passed", "violations": [], "warnings": [], "checked_claim_ids": []}}}

    def nodes(self):
        return {"technical": self.perspective("technical"), "market": self.perspective("market"),
                "stakeholder": self.perspective("stakeholder"), "domain": self.perspective("domain"),
                "synthesis": self.synthesis, "report": self.report}


def run(harness, *, supervisor=None, **state_kw):
    app = build_graph(**harness.nodes(), supervisor=supervisor, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t"}, "recursion_limit": 50}
    final = app.invoke(new_state(**state_kw), config)
    return final, app, config


def path(final):
    return [(d["action"], tuple(d["targets"])) for d in final["decision_log"]]


class RoutingTest(unittest.TestCase):
    def test_normal_run_goes_technical_then_parallel_then_stages(self):
        h = Harness()
        final, _, _ = run(h)
        self.assertEqual(path(final), [("dispatch", ("technical",)), ("dispatch", ("market", "stakeholder", "domain")),
                                       ("synthesis", ()), ("report", ()), ("quality", ()), ("finish", ())])
        self.assertEqual(final["final_status"], "ok")
        self.assertTrue(final["quality_verdict"]["passed"])
        self.assertTrue(all(d["source"] == "rule" for d in final["decision_log"]))
        self.assertEqual(h.calls, {"technical": 1, "market": 1, "stakeholder": 1, "domain": 1, "synthesis": 1, "report": 1})

    def test_insufficient_perspective_gets_exactly_one_targeted_rework(self):
        h = Harness({"stakeholder": [1, 3]})
        final, _, _ = run(h)
        self.assertEqual(h.calls["stakeholder"], 2)
        self.assertTrue(all(h.calls[p] == 1 for p in ("technical", "market", "domain")))
        self.assertIn(("dispatch", ("stakeholder",)), path(final))
        hint = h.inputs["stakeholder"][1].get("rework_hint")
        self.assertEqual(hint["requested_by"], "supervisor-sufficiency")
        self.assertEqual(h.inputs["stakeholder"][1]["request"]["max_search_rounds"], 2)  # 요청한 만큼 라운드 증가
        self.assertIsNone(final["rework_requests"]["stakeholder"])  # 처리한 요청은 비워짐
        self.assertEqual(final["final_status"], "ok")

    def test_routing_differs_between_scenarios(self):
        normal, _, _ = run(Harness())
        rework, _, _ = run(Harness({"market": [0, 3]}))
        self.assertNotEqual(path(normal), path(rework))

    def test_perspective_that_stays_insufficient_exhausts_budget_and_degrades(self):
        h = Harness({"stakeholder": [1]})
        final, _, _ = run(h)
        self.assertEqual(h.calls["stakeholder"], 3)  # 최초 + 재작업 2회
        self.assertEqual(final["final_status"], "degraded")
        self.assertTrue(any("stakeholder" in r for r in final["run_meta"]["supervisor"]["degraded_reasons"]))
        self.assertIn("report", [a for a, _ in path(final)])  # 한계를 안고도 보고서까지 간다

    def test_worker_exception_is_recorded_and_retried(self):
        h = Harness({"domain": ["raise", 3]})
        final, _, _ = run(h)
        self.assertEqual(h.calls["domain"], 2)
        self.assertEqual(final["node_status"]["domain"]["attempts"], 2)
        self.assertEqual(final["node_status"]["domain"]["status"], "ok")
        self.assertEqual(final["final_status"], "ok")

    def test_always_failing_worker_still_terminates(self):
        h = Harness({"market": ["raise"]})
        final, _, _ = run(h)
        self.assertEqual(h.calls["market"], 3)
        self.assertEqual(final["market_findings"]["status"], "failed")
        self.assertIn("RuntimeError", final["node_status"]["market"]["error"])
        self.assertEqual(final["final_status"], "degraded")
        self.assertTrue(final["last_error"])


class QualityLoopTest(unittest.TestCase):
    def test_quality_failure_loops_back_then_passes(self):
        bad_then_good = lambda n: ("# SUMMARY\n이 기술을 추천한다. 〔근거: ev:technical:0, ev:market:0, ev:stakeholder:0, ev:domain:0〕\n# REFERENCE\n" if n == 1
                                   else "# SUMMARY\n관점별 평가. 〔근거: ev:technical:0, ev:market:0, ev:stakeholder:0, ev:domain:0〕\n# REFERENCE\n")
        h = Harness(report_text=bad_then_good)
        final, _, _ = run(h)
        self.assertEqual(h.calls["report"], 2)
        self.assertEqual(h.calls["synthesis"], 1)  # 보고서만 고치면 되는 미달은 평가 종합을 다시 돌리지 않는다
        self.assertEqual(final["quality_iterations"], 2)
        self.assertTrue(final["quality_verdict"]["passed"])
        self.assertIn(("quality_rework", ("report",)), path(final))
        self.assertEqual(final["final_status"], "ok")

    def test_quality_that_never_passes_stops_at_loop_limit(self):
        h = Harness(report_text=lambda n: "# SUMMARY\n이 기술을 추천한다.\n")
        final, _, _ = run(h)
        self.assertEqual(final["quality_iterations"], 2)
        self.assertEqual(h.calls["report"], 2)
        self.assertEqual(final["final_status"], "degraded")
        self.assertIn("groundedness", final["quality_verdict"]["failed_checks"])
        self.assertIn("neutrality", final["quality_verdict"]["failed_checks"])

    def test_bias_failure_reworks_the_named_perspective(self):
        class Skewed(Harness):
            def perspective(self, p):
                node = super().perspective(p)
                if p != "market":
                    return node

                def skewed(state):
                    out = node(state)
                    if self.calls[p] == 1:  # 충분성(근거 3건·출처 2곳)은 넘지만 한 출처가 80%인 근거
                        hosts = ["a"] * 4 + ["b"]
                        out["evidence_store"] = {f"ev:market:b{i}": {**evidence(p, i, f"{h}.example.com"), "id": f"ev:market:b{i}"} for i, h in enumerate(hosts)}
                    return out
                return skewed

        h = Skewed()
        final, _, _ = run(h)
        self.assertEqual(h.calls["market"], 2)
        self.assertIn(("quality_rework", ("market",)), path(final))
        self.assertEqual(final["final_status"], "ok")

    def test_quality_checks(self):
        node = make_quality_node()
        store = {e["id"]: e for p in ("technical", "market", "stakeholder", "domain") for e in (evidence(p, 0), evidence(p, 1), evidence(p, 2))}
        ids = "ev:technical:0, ev:market:0, ev:stakeholder:0, ev:domain:0"
        ok = node({"report_sections": {"final_markdown": f"# SUMMARY\n관점별 평가 〔근거: {ids}〕\n# REFERENCE\n"}, "evidence_store": store})["quality_verdict"]
        self.assertTrue(ok["passed"])
        negated = node({"report_sections": {"final_markdown": f"# SUMMARY\n특정 기술을 추천하지 않는다. 〔근거: {ids}〕\n# REFERENCE\n"}, "evidence_store": store})["quality_verdict"]
        self.assertTrue(negated["passed"])
        bad = node({"report_sections": {"final_markdown": "# SUMMARY\nA 기술이 승자다. 〔근거: ev:nope〕\n# REFERENCE\n"}, "evidence_store": store})["quality_verdict"]
        self.assertEqual(set(bad["failed_checks"]), {"groundedness", "neutrality", "coverage"})
        missing = node({"report_sections": {"final_markdown": "# SUMMARY\n〔근거: ev:technical:0〕\n# REFERENCE\n"}, "evidence_store": store})["quality_verdict"]
        self.assertEqual(missing["failed_checks"], ["coverage"])
        self.assertEqual(missing["target_perspectives"], ["report"])


class GroundednessTest(unittest.TestCase):
    """groundedness 는 '출처로 추적되는가'를 본다: 근거 ID · 근거가 확정된 주장 ID · 코퍼스 문서 ID 는 통과, 확정되지 않은 청크 ID 는 실패."""

    def state(self):
        store = {e["id"]: {**e, "doc_id": "arxiv:1.0v1"} for e in (evidence("technical", 0), evidence("market", 0))}
        return {"evidence_store": store, "corpus_manifest": [{"doc_id": "arxiv:1.0v1"}, {"doc_id": "arxiv:2.0v1"}],
                "technical_findings": {"claims": [{"claim_id": "technical:claim:ok", "evidence_ids": ["ev:technical:0"]},
                                                   {"claim_id": "technical:claim:orphan", "evidence_ids": ["ev:gone:0"]}]},
                "synthesis": {"summary_claims": [{"claim_id": "synthesis:claim:1", "evidence_ids": ["ev:market:0"]}]}}

    def check(self, text):
        from graph.quality import check_groundedness
        return check_groundedness(self.state(), text)[0]

    def test_evidence_claim_and_document_ids_are_traceable(self):
        self.assertEqual(self.check("〔근거: ev:technical:0〕 〔근거: technical:claim:ok, synthesis:claim:1〕 〔근거: arxiv:2.0v1〕"), [])

    def test_unconfirmed_chunk_and_orphan_claim_ids_fail(self):
        issues = self.check("〔근거: technical:chunk:abc123, technical:claim:orphan, ev:technical:0〕")
        self.assertEqual(len(issues), 1)
        self.assertIn("technical:chunk:abc123", issues[0]); self.assertIn("technical:claim:orphan", issues[0])
        self.assertNotIn("ev:technical:0", issues[0])

    def test_malformed_closing_bracket_does_not_glue_ids_together(self):
        text = "〔근거: ev:technical:0】.\n\n〔근거: ev:market:0〕"
        self.assertEqual(self.check(text), [])

    def test_no_citation_at_all_fails(self):
        self.assertIn("인용이 하나도 없음", self.check("근거 없는 보고서")[0])


class StructureCheckTest(unittest.TestCase):
    def check(self, text):
        from graph.quality import check_structure
        return check_structure({"report_sections": {"final_markdown": text}})[0]

    def test_summary_first_reference_last_passes(self):
        self.assertEqual(self.check("# SUMMARY\n짧은 요약\n\n# 1. 본문\n내용\n\n# REFERENCE\n- 출처"), [])

    def test_wrong_order_and_long_summary_fail(self):
        self.assertTrue(self.check("# 1. 본문\n내용\n\n# SUMMARY\n요약\n\n# REFERENCE\n"))
        self.assertTrue(self.check("# SUMMARY\n" + "가" * 1200 + "\n\n# 1. 본문\n내용\n\n# REFERENCE\n"))
        self.assertTrue(self.check("# SUMMARY\n요약\n\n# REFERENCE\n\n# 6. 한계점\n끝"))

    def test_no_report_means_no_structure_check(self):
        self.assertEqual(self.check(""), [])


class LengthCheckTest(unittest.TestCase):
    def verdict(self, layout):
        store = {e["id"]: e for p in ("technical", "market", "stakeholder", "domain") for e in (evidence(p, 0), evidence(p, 1), evidence(p, 2))}
        ids = "ev:technical:0, ev:market:0, ev:stakeholder:0, ev:domain:0"
        state = {"report_sections": {"final_markdown": f"# SUMMARY\n관점별 평가 〔근거: {ids}〕\n# REFERENCE\n"}, "evidence_store": store,
                 "run_meta": {"report": {"pdf_layout": layout}} if layout else {}}
        return make_quality_node()(state)["quality_verdict"]

    def test_pdf_within_ten_pages_passes(self):
        self.assertTrue(self.verdict({"pages": 9, "max_pages": 10, "within_limit": True})["passed"])

    def test_pdf_over_limit_fails_length_and_targets_report(self):
        v = self.verdict({"pages": 12, "max_pages": 10, "within_limit": False})
        self.assertEqual(v["failed_checks"], ["length"])
        self.assertEqual(v["target_perspectives"], ["report"])

    def test_no_pdf_means_no_length_check(self):
        self.assertTrue(self.verdict(None)["passed"])


class TerminationTest(unittest.TestCase):
    def test_step_cap_still_finishes_with_synthesis_and_report(self):
        h = Harness()
        final, _, _ = run(h, max_steps=2)
        actions = [a for a, _ in path(final)]
        self.assertEqual(actions[-3:], ["synthesis", "report", "finish"])
        self.assertEqual(final["final_status"], "degraded")
        self.assertTrue(any("step 상한" in r for r in final["run_meta"]["supervisor"]["degraded_reasons"]))
        self.assertLessEqual(final["step_count"], 2 + 3)

    def test_decision_log_is_capped_but_external_log_is_full(self):
        entries = [{"step": i, "action": "x", "targets": [], "reason": "", "source": "rule"} for i in range(50)]
        self.assertEqual(len(append_capped([], entries)), 20)
        self.assertEqual(append_capped([], entries)[-1]["step"], 49)

    def test_decisions_are_written_to_external_log(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            final, _, _ = run(Harness(), supervisor=make_supervisor(log_dir=tmp))
            lines = (Path(tmp) / f"{final['trace_id']}.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), len(final["decision_log"]))
            self.assertEqual(json.loads(lines[0])["run_id"], final["run_id"])

    def test_checkpoint_keeps_final_state_for_resume(self):
        final, app, config = run(Harness())
        saved = app.get_state(config).values
        self.assertEqual(saved["final_status"], final["final_status"])
        self.assertEqual(saved["step_count"], final["step_count"])


class TopologyTest(unittest.TestCase):
    def test_workers_only_return_to_supervisor(self):
        app = build_graph(**Harness().nodes())
        graph = app.get_graph()
        workers = {"technical", "market", "stakeholder", "domain", "synthesis", "report", "quality"}
        for edge in graph.edges:
            if edge.source in workers:
                self.assertEqual(edge.target, "supervisor", edge)
        self.assertTrue(any(e.conditional for e in graph.edges if e.source == "supervisor"))


class LLMGuardTest(unittest.TestCase):
    def fake_client(self, payload=None, exc=None):
        def parse(**kw):
            if exc:
                raise exc
            return SimpleNamespace(status="completed", output_parsed=SimpleNamespace(model_dump=lambda: payload))
        return SimpleNamespace(responses=SimpleNamespace(parse=parse))

    def test_valid_llm_proposal_is_used(self):
        proposer = make_llm_proposer(self.fake_client({"action": "dispatch", "targets": ["technical"], "reason": "기술 조사 먼저"}), "gpt-5.5")
        final, _, _ = run(Harness(), supervisor=make_supervisor(proposer))
        self.assertEqual(final["decision_log"][0]["source"], "llm")
        self.assertEqual(final["decision_log"][0]["reason"], "기술 조사 먼저")

    def test_illegal_llm_proposal_falls_back_to_rule(self):
        proposer = make_llm_proposer(self.fake_client({"action": "report", "targets": [], "reason": "바로 보고서"}), "gpt-5.5")
        h = Harness()
        final, _, _ = run(h, supervisor=make_supervisor(proposer))
        self.assertTrue(all(d["source"] == "fallback" for d in final["decision_log"][:2]))
        self.assertEqual(final["decision_log"][0]["action"], "dispatch")  # 근거 수집 전 보고서는 막힌다
        self.assertEqual(final["final_status"], "ok")

    def test_llm_error_falls_back_and_graph_still_finishes(self):
        proposer = make_llm_proposer(self.fake_client(exc=RuntimeError("model not found")), "gpt-5.5")
        final, _, _ = run(Harness(), supervisor=make_supervisor(proposer))
        self.assertEqual({d["source"] for d in final["decision_log"]}, {"fallback"})
        self.assertIn("LLM 호출 실패", final["decision_log"][0]["reason"])
        self.assertEqual(final["final_status"], "ok")

    def test_llm_cannot_pick_target_outside_candidates(self):
        state = new_state()
        opts = options(state)
        rule = rule_choice(state, opts)
        decision, source = guard({"action": "dispatch", "targets": ["market"], "reason": "x"}, opts, rule, False)
        self.assertEqual((decision["targets"], source), (["technical"], "fallback"))

    def test_sequential_mode_dispatches_one_worker_per_turn(self):
        final, _, _ = run(Harness(), supervisor=make_supervisor(sequential=True))
        dispatches = [t for a, t in path(final) if a == "dispatch"]
        self.assertTrue(all(len(t) == 1 for t in dispatches))
        self.assertEqual(len(dispatches), 4)


if __name__ == "__main__":
    unittest.main()


class StateContractTest(unittest.TestCase):
    """README 의 State Schema 7항목이 코드에서 실제로 성립하는지."""

    def test_trace_id_is_shared_by_state_log_and_config(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            state = new_state(run_id="abc123")
            app = build_graph(**Harness().nodes(), supervisor=make_supervisor(log_dir=tmp), checkpointer=InMemorySaver())
            config = {"configurable": {"thread_id": state["run_id"]}, "metadata": {"trace_id": state["trace_id"]}, "recursion_limit": 50}
            final = app.invoke(state, config)
            lines = [json.loads(line) for line in (Path(tmp) / f"{final['trace_id']}.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(final["trace_id"], "trace-abc123")
        self.assertTrue(all(entry["trace_id"] == "trace-abc123" and entry["run_id"] == "abc123" for entry in lines))
        self.assertEqual(app.get_state(config).config["configurable"]["thread_id"], "abc123")

    def test_conflicting_concurrent_evidence_writes_keep_existing_and_fill_blanks(self):
        from graph.state import merge_evidence_store
        old = evidence("market", 0)
        new = {**old, "quote": "다른 인용", "title": "", "url": None, "metric_tag": "tag"}
        merged = merge_evidence_store({old["id"]: {**old, "metric_tag": None}}, {old["id"]: new})[old["id"]]
        self.assertEqual(merged["quote"], old["quote"])      # 기존 값 우선
        self.assertEqual(merged["metric_tag"], "tag")        # 빈 필드만 보충
        self.assertEqual(merge_evidence_store(None, None), {})

    def test_sqlite_checkpoint_survives_a_new_process_graph(self):
        try:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError:
            self.skipTest("langgraph-checkpoint-sqlite 미설치")
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "cp.sqlite")
            config = {"configurable": {"thread_id": "resume-1"}, "recursion_limit": 50}
            first = build_graph(**Harness().nodes(), checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
            final = first.invoke(new_state(run_id="resume-1"), config)
            second = build_graph(**Harness().nodes(), checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
            saved = second.get_state(config).values  # 새 그래프 객체가 파일에서 상태를 읽는다
        self.assertEqual(saved["final_status"], final["final_status"])
        self.assertEqual(saved["node_status"]["report"], final["node_status"]["report"])
        self.assertEqual(saved["step_count"], final["step_count"])


class OfflineMainTest(unittest.TestCase):
    def test_offline_main_produces_report_and_terminates(self):
        import subprocess
        import tempfile
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            done = subprocess.run([sys.executable, "main.py", "--no-pdf", "--output-dir", tmp], cwd=root, capture_output=True, text=True, timeout=120)
            folders = list(Path(tmp).iterdir())
            self.assertEqual(len(folders), 1, done.stdout + done.stderr)
            report = (folders[0] / "report.md").read_text(encoding="utf-8")
            summary = (folders[0] / "summary.md").read_text(encoding="utf-8")
        self.assertTrue(report.lstrip().startswith("# SUMMARY") and "# REFERENCE" in report)
        self.assertIn("## Supervisor", summary)
        self.assertIn("supervisor: 종료 상태", done.stdout)
