"""평가 종합 에이전트 테스트. LLM·네트워크 없이 합성 fixture 로 검사한다.

실행: python -m unittest tests.agents.synthesis.test_synthesis -v   (pytest 로도 동작)
"""

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agents.synthesis import make_node  # noqa: E402
from agents.synthesis.matrix import build_matrix  # noqa: E402
from agents.synthesis.node import project_input  # noqa: E402
from agents.synthesis.review import check_claim, review_claims, review_context  # noqa: E402
from agents.synthesis.rules import condition_violations, numbers_in  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "appstate_sample.json"


def load():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def run(state, writer=None):
    return make_node(writer)(state)["synthesis"]


def rules_of(result):
    return {f["rule_id"] for f in result["cross_findings"] if f["kind"] == "conflict"}


class FixedWriter:
    """정해진 문장을 돌려주는 가짜 LLM. revise 결과도 지정할 수 있다."""

    name, model = "fixed", None

    def __init__(self, claims, revised=None, explanations=None):
        self.claims, self.revised, self.explanations = claims, revised, explanations or {}
        self.revise_calls = 0

    def write(self, payload):
        return {"claims": copy.deepcopy(self.claims), "explanations": self.explanations}

    def revise(self, claim, violations, payload):
        self.revise_calls += 1
        return copy.deepcopy(self.revised)


class NodeContractTest(unittest.TestCase):
    def test_returns_only_synthesis_key(self):
        out = make_node()(load())
        self.assertEqual(list(out), ["synthesis"])
        for key in ("status", "as_of", "input_hash", "matrix", "cross_findings", "contrast_table", "summary_claims",
                    "gaps", "imbalance", "retry_requests", "limitations", "dropped_sentences", "meta"):
            self.assertIn(key, out["synthesis"])

    def test_no_internal_fields_leak(self):
        for f in run(load())["cross_findings"]:
            self.assertFalse([k for k in f if k.startswith("_")])

    def test_projection_reads_only_needed_keys(self):
        local = project_input(load())
        self.assertEqual(set(local), {"as_of", "tech_names", "findings", "evidence_store"})
        self.assertEqual(local["tech_names"], {"sw": "MLA", "hw": "ITME"})

    def test_all_perspectives_missing_is_failed_not_exception(self):
        state = load()
        for p in ("technical", "market", "stakeholder", "domain"):
            state[f"{p}_findings"] = None
        result = run(state)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["summary_claims"], [])
        self.assertEqual(len(result["retry_requests"]), 4)

    def test_writer_exception_is_recorded(self):
        class Broken(FixedWriter):
            def write(self, payload):
                raise RuntimeError("boom")
        result = run(load(), Broken([]))
        self.assertTrue(any("서술 단계 실패" in x for x in result["limitations"]))
        self.assertNotEqual(result["status"], "complete")

    def test_works_inside_parent_graph(self):
        from graph.build import build_graph
        state = load()
        stub = lambda key: (lambda s: {key: state[key]})
        app = build_graph(technical=stub("technical_findings"), market=stub("market_findings"),
                          stakeholder=stub("stakeholder_findings"), domain=stub("domain_findings"),
                          synthesis=make_node(), report=lambda s: {"report_sections": {}})
        empty = {**state, **{f"{p}_findings": None for p in ("technical", "market", "stakeholder", "domain")}}
        out = app.invoke(empty)
        self.assertIsNotNone(out["synthesis"])
        self.assertGreater(len(out["synthesis"]["matrix"]), 0)


class MatrixTest(unittest.TestCase):
    def test_matrix_cell_per_record(self):
        result = run(load())
        self.assertEqual(len(result["matrix"]), 12)

    def test_n_evidence_counts_distinct_content_hash(self):
        state = load()
        state["technical_findings"]["records"][1]["evidence_ids"].append("domain:ev:001")  # technical:ev:002 와 같은 hash
        cell = next(c for c in run(state)["matrix"] if c["perspective"] == "technical" and c["technology"] == "hw")
        self.assertEqual(cell["n_evidence"], 2)

    def test_missing_perspective_becomes_gap_and_retry(self):
        state = load()
        state["market_findings"] = None
        result = run(state)
        self.assertEqual(result["status"], "partial")
        self.assertIn({"perspective": "market", "reason": "결과 없음"}, result["retry_requests"])
        self.assertTrue(any(g["perspective"] == "market" for g in result["gaps"]))

    def test_unknown_evidence_id_is_excluded_and_logged(self):
        state = load()
        state["domain_findings"]["records"][0]["evidence_ids"].append("ghost:ev:999")
        result = run(state)
        self.assertTrue(any("evidence_store 에 없는 근거" in x for x in result["limitations"]))

    def test_not_applicable_excluded_from_imbalance(self):
        basis = run(load())["imbalance"]["basis_counts"]
        self.assertNotIn("not_applicable", basis["sw"])

    def test_imbalance_flagged_at_two_times(self):
        state = load()
        for p in ("technical", "market", "stakeholder", "domain"):
            state[f"{p}_findings"]["records"] = [r for r in state[f"{p}_findings"]["records"] if r["technology"] == "hw"]
        state["technical_findings"]["records"].append({**load()["technical_findings"]["records"][0]})
        result = run(state)
        self.assertTrue(result["imbalance"]["flagged"])

    def test_future_evidence_is_flagged(self):
        state = load()
        state["evidence_store"]["domain:ev:002"]["published_at"] = "2027-01-01"
        self.assertTrue(any("기준일 이후" in x for x in build_matrix(project_input(state))["limitations"]))


class RelationsTest(unittest.TestCase):
    def test_fixture_triggers_all_conflict_rules(self):
        self.assertEqual(rules_of(run(load())), {"SX1", "SX2", "SX3", "SX4", "SX5", "SX6", "SX7"})

    def test_shared_evidence_is_not_agreement(self):
        result = run(load())
        shared = [f for f in result["cross_findings"] if f["kind"] == "shared_evidence"]
        refs = {tuple(f["record_refs"]) for f in shared}
        self.assertIn(("domain/sw/HBM 용량 압박 완화", "technical/sw/성숙도(TRL)"), refs)
        agreement_refs = {tuple(f["record_refs"]) for f in result["cross_findings"] if f["kind"] == "agreement"}
        self.assertFalse(refs & agreement_refs)

    def test_different_vocab_same_direction_is_complement(self):
        kinds = {(f["kind"], tuple(f["record_refs"])) for f in run(load())["cross_findings"]}
        self.assertIn(("complement", ("market/sw/상용화·채택 현황", "technical/sw/성숙도(TRL)")), kinds)

    def test_same_vocab_same_direction_is_agreement(self):
        state = load()
        extra = copy.deepcopy(state["market_findings"]["records"][0])
        extra.update(perspective="stakeholder", criterion="도입 기업·개발자(시장 어휘)", evidence_ids=["stakeholder:ev:001"])
        state["stakeholder_findings"]["records"].append(extra)
        self.assertTrue(any(f["kind"] == "agreement" for f in run(state)["cross_findings"]))

    def test_sx1_needs_trl_below_seven(self):
        state = load()
        state["technical_findings"]["records"][1]["value"] = "7-8"
        hw_sx1 = [f for f in run(state)["cross_findings"] if f["rule_id"] == "SX1" and f["technology"] == "hw"]
        self.assertEqual(hw_sx1, [])

    def test_unknown_basis_does_not_trigger_sx2(self):
        refs = [r for f in run(load())["cross_findings"] if f["rule_id"] == "SX2" for r in f["record_refs"]]
        self.assertNotIn("domain/hw/전력·발열", refs)
        self.assertNotIn("stakeholder/hw/투자·산업 관계자", refs)

    def test_order_swap_gives_same_result(self):
        """제시 순서 편향 방지: 입력 dict 순서를 바꿔도 매트릭스·관계가 같아야 한다 (§6.4)."""
        a = load()
        b = load()
        b["selected_tech"] = {"hw": b["selected_tech"]["hw"], "sw": b["selected_tech"]["sw"]}
        for p in ("technical", "market", "stakeholder", "domain"):
            b[f"{p}_findings"]["records"].reverse()
            b[f"{p}_findings"]["claims"].reverse()
        b["evidence_store"] = dict(reversed(list(b["evidence_store"].items())))
        ra, rb = run(a), run(b)
        self.assertEqual(ra["matrix"], rb["matrix"])
        self.assertEqual(ra["cross_findings"], rb["cross_findings"])
        self.assertEqual(ra["contrast_table"], rb["contrast_table"])

    def test_contrast_unknown_when_one_side_missing(self):
        rows = {r["criterion"]: r for r in run(load())["contrast_table"]}
        self.assertEqual(rows["technical/성숙도(TRL)"]["relation"], "independent")
        self.assertEqual(rows["domain/전력·발열"]["relation"], "unknown")


class ReviewTest(unittest.TestCase):
    def ctx(self):
        return review_context(project_input(load()))

    def claim(self, text, ids=("technical:ev:001",), tech="sw"):
        return {"technology": tech, "text": text, "evidence_ids": list(ids), "conditions": [], "limitations": []}

    def test_clean_sentence_passes(self):
        self.assertEqual(check_claim(self.claim("MLA는 DeepSeek 67B 대비 KV cache를 93.3% 줄였다."), self.ctx()), [])

    def test_c1_unknown_evidence(self):
        self.assertTrue(any(v.startswith("C1") for v in check_claim(self.claim("근거 없는 문장", ids=("ghost",)), self.ctx())))

    def test_c3_number_not_in_evidence(self):
        self.assertTrue(any(v.startswith("C3") for v in check_claim(self.claim("KV cache를 50% 줄였다."), self.ctx())))

    def test_c4_condition_and_forbidden_numbers(self):
        self.assertTrue(condition_violations("ITME 처리량이 35.7% 향상됐다.", "hw"))
        self.assertFalse(condition_violations("ITME는 CPU-offload 대비 최대 35.7% 처리량 향상을 보였다.", "hw"))
        self.assertTrue(condition_violations("MLA는 훈련비를 42.5% 줄였다.", "sw"))

    def test_c5_ranking_words(self):
        self.assertTrue(any(v.startswith("C5") for v in check_claim(self.claim("MLA가 ITME보다 우수하다."), self.ctx())))

    def test_c6_proxy_needs_class_label(self):
        v = check_claim(self.claim("CXL 시장이 커지고 있다.", ids=("market:ev:005",), tech="hw"), self.ctx())
        self.assertTrue(any("기술군" in x for x in v))

    def test_c7_trl_must_match_matrix(self):
        v = check_claim(self.claim("ITME의 TRL 7로 판단된다.", ids=("technical:ev:002",), tech="hw"), self.ctx())
        self.assertTrue(any(x.startswith("C7") for x in v))

    def test_numbers_ignore_ids_and_product_names(self):
        self.assertEqual(numbers_in("market:claim:003, DeepSeek-V2, 8×H800, 67B"), ["8"])

    def test_violation_revised_once_then_kept(self):
        bad = self.claim("MLA가 압도적이다.")
        good = self.claim("MLA는 DeepSeek 67B 대비 KV cache를 93.3% 줄였다.")
        writer = FixedWriter([bad], revised=good)
        result = review_claims([bad], {}, self.ctx(), writer, {})
        self.assertEqual(writer.revise_calls, 1)
        self.assertEqual(result["claims"], [good])
        self.assertEqual(result["dropped"], [])

    def test_violation_still_bad_after_revise_is_dropped(self):
        bad = self.claim("MLA가 압도적이다.")
        writer = FixedWriter([bad], revised=bad)
        result = review_claims([bad], {}, self.ctx(), writer, {})
        self.assertEqual(writer.revise_calls, 1)
        self.assertEqual(result["claims"], [])
        self.assertEqual(result["dropped"][0]["stage"], "summary_claim")

    def test_end_to_end_drop_is_reported(self):
        writer = FixedWriter([self.claim("MLA가 ITME보다 우수하다.")], revised=None)
        result = run(load(), writer)
        self.assertEqual(result["summary_claims"], [])
        self.assertEqual(len(result["dropped_sentences"]), 1)
        self.assertEqual(result["meta"]["quality"]["status"], "failed")

    def test_explanation_sets_resolution(self):
        conflict_id = next(f["id"] for f in run(load())["cross_findings"] if f["rule_id"] == "SX1")
        writer = FixedWriter([self.claim("MLA는 DeepSeek 67B 대비 KV cache를 93.3% 줄였다.")],
                             explanations={conflict_id: ("시장 신호는 기술군 발표이고 TRL 은 논문 근거라 범위가 다르다.", True)})
        finding = next(f for f in run(load(), writer)["cross_findings"] if f["id"] == conflict_id)
        self.assertEqual(finding["resolution"], "explained")

    def test_template_output_passes_review(self):
        result = run(load())
        self.assertEqual(result["dropped_sentences"], [])
        self.assertEqual(result["meta"]["quality"]["status"], "passed")
        self.assertTrue(all(c["claim_id"].startswith("synthesis:claim:") for c in result["summary_claims"]))


class OpenAIWriterTest(unittest.TestCase):
    """실제 API 없이 OpenAIWriter 의 요청·응답 처리만 검사한다 (가짜 client)."""

    def make(self, status="completed", parsed=None):
        from types import SimpleNamespace
        from agents.synthesis.writer import OpenAIWriter
        calls = []

        def parse(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(status=status, output_parsed=parsed, incomplete_details={"reason": "max_output_tokens"})
        client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
        return OpenAIWriter(model="test-model", client=client), calls

    def test_write_sends_quotes_not_full_documents(self):
        from agents.synthesis.schemas import Draft, DraftClaim, Explanation
        draft = Draft(claims=[DraftClaim(technology="sw", text="MLA는 DeepSeek 67B 대비 KV cache를 93.3% 줄였다.",
                                         evidence_ids=["technical:ev:001"], conditions=[], limitations=[])],
                      explanations=[Explanation(finding_id="sx1:hw:001", explanation="범위가 다르다.", resolved=True)])
        writer, calls = self.make(parsed=draft)
        result = run(load(), writer)
        self.assertEqual(len(result["summary_claims"]), 1)
        self.assertEqual(calls[0]["model"], "test-model")
        self.assertIn("93.3%", calls[0]["input"])  # quote 는 들어감
        self.assertNotIn("page_or_locator", calls[0]["input"])  # 원문 메타 전체는 넣지 않음
        self.assertEqual(result["meta"]["writer"], "openai")

    def test_incomplete_response_is_recorded_not_raised(self):
        writer, _ = self.make(status="incomplete")
        result = run(load(), writer)
        self.assertTrue(any("응답 미완료" in x for x in result["limitations"]))


if __name__ == "__main__":
    unittest.main()
