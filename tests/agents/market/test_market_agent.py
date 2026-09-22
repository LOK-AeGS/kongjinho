"""시장 평가 에이전트 계약 테스트. LLM·네트워크 없이 순수 로직 + 가짜 LLM으로 검사한다.

실행: python tests/agents/market/test_market_agent.py   (pytest 로도 동작)
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agents.market.node import make_node, project_input, summarize_technical  # noqa: E402
from agents.market.prompts import JUDGE_SYSTEM, CompareOut, CounterCheck, InferenceItem, Judgement, PlanOut  # noqa: E402
from agents.market.quality import linter, rubric  # noqa: E402
from agents.market.rag import tier  # noqa: E402
from agents.market.rag.evidence import make_evidence_id, normalize_url  # noqa: E402
from agents.market.rag.fetch import DocumentPart, FetchedDocument  # noqa: E402
from agents.market.rag.index import chunk_document  # noqa: E402
from agents.market.rag.retriever import fake_retriever, make_rag_fn  # noqa: E402
from agents.market.state import new_evidence, to_perspective_findings  # noqa: E402
from agents.market.subgraph import MarketAgentDeps, _collect_body  # noqa: E402

AS_OF = "2026-09-22"
SW, HW = "DeepSeek-V2 (MLA)", "ITME (CXL-Hybrid)"

# --- 근거 ID -----------------------------------------------------------------


def test_같은_출처는_다시_가져와도_같은_ID를_받는다():
    a = make_evidence_id("https://arxiv.org/pdf/2405.04434", "p.21", "reduces the KV cache by 93.3%")
    b = make_evidence_id("https://ARXIV.org/pdf/2405.04434/", "p.21", "reduces the KV cache by  93.3% ")
    assert a == b


def test_추적_파라미터는_ID에_영향을_주지_않는다():
    assert normalize_url("https://www.vendor.com/a?utm_source=x") == "https://vendor.com/a"


def test_다른_인용문은_다른_ID를_받는다():
    a = make_evidence_id("https://x.com", "p.1", "첫 번째 문장")
    b = make_evidence_id("https://x.com", "p.1", "두 번째 문장")
    assert a != b


# --- Rubric 판정 --------------------------------------------------------------


def _claim(cid, tid, aspect, level, statement, ev_id, stance="support", scope="direct", basis="direct_evidence"):
    return {"claim_id": cid, "technology_id": tid, "aspect": aspect, "basis": basis, "evidence_level": level,
            "statement": statement, "evidence_ids": [ev_id] if ev_id else [], "stance": stance, "scope": scope,
            "observed_at": "2026-08", "uncertainty": "u", "is_stub": False}


def _ev(eid, url, source_type="news", doc_id=None):
    return {"id": eid, "url": url, "locator": url, "title": "t", "organization": "org", "source_type": source_type,
            "published_at": "2026-08", "accessed_at": "2026-08-01T00:00:00Z", "quote": "q", "doc_id": doc_id, "scope": "direct"}


def test_시장규모는_서로_다른_출처_2건과_수치가_있어야_충분이다():
    names = {"sw": SW, "hw": HW}
    c2 = [_claim("c1", "sw", "size", "forecast", "2025년 10억 달러", "e1"), _claim("c2", "sw", "size", "forecast", "CAGR 20%", "e2")]
    e2 = [_ev("e1", "https://a.com/x"), _ev("e2", "https://b.com/y")]
    v = next(v for v in rubric.compute_verdicts(c2, e2) if v["technology_id"] == "sw" and v["criterion"] == "시장 규모·성장성")
    assert v["verdict"] == "충분"
    v1 = next(v for v in rubric.compute_verdicts(c2[:1], e2) if v["technology_id"] == "sw" and v["criterion"] == "시장 규모·성장성")
    assert v1["verdict"] == "부분"
    v0 = next(v for v in rubric.compute_verdicts([], []) if v["technology_id"] == "sw" and v["criterion"] == "시장 규모·성장성")
    assert v0["verdict"] == "부족"


def test_블로그_단독_근거는_충분으로_올리지_않는다():
    claims = [_claim("c1", "sw", "size", "forecast", "2025년 10억", "e1"), _claim("c2", "sw", "size", "forecast", "CAGR 20%", "e2")]
    evs = [_ev("e1", "https://a.com/x", "community"), _ev("e2", "https://b.com/y", "community")]
    v = next(v for v in rubric.compute_verdicts(claims, evs) if v["technology_id"] == "sw" and v["criterion"] == "시장 규모·성장성")
    assert v["verdict"] == "부분" and any("블로그" in x for x in v["limitations"])


def test_논문_근거는_실운용_채택으로_인정하지_않는다():
    claim = [_claim("c1", "hw", "adoption", "production", "s", "e1")]
    ev_paper = [_ev("e1", "https://arxiv.org/abs/1", "paper", doc_id="itme")]
    ev_official = [_ev("e1", "https://vendor.com/p", "official_web")]
    v_paper = next(v for v in rubric.compute_verdicts(claim, ev_paper) if v["technology_id"] == "hw" and v["criterion"] == "상용화·채택 현황")
    v_official = next(v for v in rubric.compute_verdicts(claim, ev_official) if v["technology_id"] == "hw" and v["criterion"] == "상용화·채택 현황")
    assert v_paper["verdict"] == "부분" and v_official["verdict"] == "충분"


def test_반대_근거가_없는_기술은_counter_missing에_잡힌다():
    claims = [_claim("c1", "sw", "size", "forecast", "s", "e1")]
    assert rubric.counter_missing(claims) == ["sw", "hw"]
    with_counter = claims + [_claim("c2", "sw", "counter", "unknown", "s", "e2", stance="counter")]
    assert rubric.counter_missing(with_counter) == ["hw"]


def test_근거_수_불균형을_감지한다():
    names = {"sw": SW, "hw": HW}
    many = [_claim(f"c{i}", "sw", "size", "unknown", "s", f"e{i}") for i in range(6)]
    few = [_claim("cx", "hw", "size", "unknown", "s", "ex")]
    assert rubric.imbalance_note(many + few, names) is not None
    assert rubric.imbalance_note(many[:2] + few, names) is None


# --- 표현 린터 -----------------------------------------------------------------


def test_판단_프롬프트는_이름이_비슷한_별개_기술을_경고한다():
    """2026-09-22 실 API 실행에서 SK hynix IMTE를 대상 기술 ITME로 오귀속할 뻔한 사례(§JUDGE_SYSTEM 규칙 2).

    LLM이 실제로 이 규칙을 지키는지는 실 API로만 확인 가능하다. 여기서는 규칙 문구 자체가
    프롬프트에서 조용히 빠지지 않는지만 회귀 검사한다.
    """
    assert "발행 주체" in JUDGE_SYSTEM and "다른 별개 기술" in JUDGE_SYSTEM


def test_우열_어휘를_잡는다():
    hits = linter.lint_statements([("c1", "A가 B보다 더 우수하고 압도적이다"), ("c2", "정상적인 사실 서술")])
    assert {h["claim_id"] for h in hits} == {"c1"}


# --- 출처 등급 필터 (Pool B) ---------------------------------------------------


def test_등급표에_없는_도메인은_기타로_분류되고_걸러진다():
    assert tier.classify("https://arxiv.org/abs/1") == "paper"
    assert tier.classify("https://www.nvidia.com/blog/x") == "vendor"
    assert tier.classify("https://random-blog.example.net/post") == "other"
    kept, dropped = tier.filter_trusted([{"url": "https://arxiv.org/abs/1"}, {"url": "https://random-blog.example.net/post"}])
    assert len(kept) == 1 and len(dropped) == 1
    assert "기타" not in dropped[0] and "등급표에 없어" in dropped[0]


def test_등급표는_실행에서_부당하게_걸러졌던_정당한_출처를_포함한다():
    """2026-09-22 실 API 실행 3회에서 걸러진 도메인 중 정당한 출처였던 것들."""
    assert tier.classify("https://api-docs.deepseek.com/quick_start") == "vendor"  # 대상 SW 원저작사
    assert tier.classify("https://www.databricks.com/blog/x") == "vendor"
    assert tier.classify("https://www.redhat.com/en/blog/x") == "vendor"
    assert tier.classify("https://www.mordorintelligence.com/industry-reports/x") == "research"
    assert tier.classify("https://www.trendforce.com/news/x") == "research"
    assert tier.classify("https://www.alphaxiv.org/abs/2606.12556") == "paper"
    assert tier.classify("https://ar5iv.labs.arxiv.org/html/2606.12556") == "paper"  # arxiv.org 하위 도메인으로 이미 포함
    assert tier.classify("https://www.semanticscholar.org/paper/ITME") == "paper"  # 대상 HW 논문 원문 링크가 걸렸던 사례


# --- 본문 수집·청킹·색인 (Pool B) ----------------------------------------------


def _short_doc(url="https://nvidia.com/a"):
    return FetchedDocument(url=url, title="짧은 문서", parts=[DocumentPart(text="본문이 짧다.", locator=url)], page_count=1, is_long=False)


def _long_doc(url="https://arxiv.org/abs/1"):
    text_a = ("KV cache 압축과 데이터센터 채택 관련 내용. " * 50)
    text_b = ("이 기술의 확산에는 비용과 재학습이라는 걸림돌이 있다. " * 50)
    return FetchedDocument(
        url=url, title="긴 문서",
        parts=[DocumentPart(text=text_a, locator="p.1"), DocumentPart(text=text_b, locator="p.2")],
        page_count=2, is_long=True,
    )


def test_짧은_문서는_청킹_없이_그대로_근거_후보가_된다():
    def fetch_body(url):
        return _short_doc(url)

    deps = MarketAgentDeps(llm=None, web_search=None, fetch_body=fetch_body, page_budget=200)
    results = [{"url": "https://nvidia.com/a", "title": "t", "content": "스니펫", "organization": "nvidia.com",
                "published_date": None, "source_type": "official_web"}]
    out, pages_used, log = _collect_body(results, "query", deps, pages_used=0)
    assert len(out) == 1 and out[0]["content"] == "본문이 짧다."
    assert pages_used == 1 and not log


def test_긴_문서는_청킹_색인_후_질의에_맞는_조각만_근거_후보가_된다():
    doc = _long_doc()
    chunks = chunk_document(doc)
    assert len(chunks) > 2  # 두 part가 CHUNK_CHARS를 넘겨 여러 조각으로 쪼개졌다

    def fetch_body(url):
        return doc

    deps = MarketAgentDeps(llm=None, web_search=None, fetch_body=fetch_body, page_budget=200)
    results = [{"url": doc.url, "title": "t", "content": "스니펫", "organization": "arxiv.org",
                "published_date": None, "source_type": "paper"}]
    out, pages_used, log = _collect_body(results, "비용과 재학습 걸림돌", deps, pages_used=0)
    assert 0 < len(out) <= 3
    assert any("걸림돌" in r["content"] or "재학습" in r["content"] for r in out)
    assert pages_used == 2


def test_페이지_예산을_넘으면_본문_수집을_중단한다():
    def fetch_body(url):
        return _short_doc(url)

    deps = MarketAgentDeps(llm=None, web_search=None, fetch_body=fetch_body, page_budget=1)
    results = [{"url": f"https://nvidia.com/{i}", "title": "t", "content": "s", "organization": "nvidia.com",
                "published_date": None, "source_type": "official_web"} for i in range(3)]
    out, pages_used, log = _collect_body(results, "q", deps, pages_used=1)  # 이미 한도 도달
    assert out == [] and pages_used == 1
    assert any("한도" in m["message"] for m in log)


def test_등급표에_없는_출처는_본문_수집_전에_걸러진다():
    def fetch_body(url):
        raise AssertionError("걸러진 URL은 fetch_body가 호출되면 안 된다")

    deps = MarketAgentDeps(llm=None, web_search=None, fetch_body=fetch_body, page_budget=200)
    results = [{"url": "https://random-blog.example.net/post", "title": "t", "content": "s",
                "organization": "x", "published_date": None, "source_type": "other"}]
    out, pages_used, log = _collect_body(results, "q", deps, pages_used=0)
    assert out == [] and pages_used == 0 and log


# --- State 변환 (AppState) -----------------------------------------------------


def _sample_result():
    ev = new_evidence(url="https://a.com/x", locator="https://a.com/x", title="t", organization="org",
                       source_type="news", published_at="2026-08", accessed_at="2026-08-01T00:00:00Z",
                       quote="시장이 성장한다", doc_id=None, scope="direct")
    claim = _claim("market:claim:001", "sw", "size", "forecast", "시장이 성장한다", ev["id"])
    completion = {"status": "complete", "search_rounds_used": 1, "revision_rounds_used": 0, "gaps": [], "errors": []}
    return {"claims": [claim], "evidence": [ev], "technologies": {"sw": SW, "hw": HW}}, completion


def test_v08_basis는_기존_충분_부분_부족_판정에서_그대로_유도된다():
    result, completion = _sample_result()
    result["verdicts"] = rubric.compute_verdicts(result["claims"], result["evidence"])
    out = to_perspective_findings(result, completion)
    size = next(r for r in out["market_findings"]["records"] if r["technology"] == "sw" and r["criterion"] == "시장 규모·성장성")
    assert size["basis"] == "inferred"  # rubric 판정 '부분' -> §2.2.4 표의 inferred
    assert size["assessment_vocab"] == "market_signal"
    assert size["assessment"] in ("adopted", "announced", "projected", "none")
    missing = next(r for r in out["market_findings"]["records"] if r["technology"] == "hw" and r["criterion"] == "상용화·채택 현황")
    assert missing["basis"] == "unknown"  # rubric 판정 '부족' -> unknown


def test_v08_변환은_evidence_store를_market_findings_밖에_별도로_둔다():
    result, completion = _sample_result()
    result["verdicts"] = rubric.compute_verdicts(result["claims"], result["evidence"])
    out = to_perspective_findings(result, completion)
    assert set(out) == {"market_findings", "evidence_store"}
    ev = next(iter(out["evidence_store"].values()))
    assert ev["id"] == result["evidence"][0]["id"]  # id를 재해시하지 않고 그대로 재사용한다
    assert ev["stance"] == "support" and ev["direct_or_proxy"] == "direct"
    assert len(ev["content_hash"]) == 64  # sha256 hex


def test_v08_변환은_부족한_칸과_반대근거_누락을_구조화된_gap으로_남긴다():
    result, completion = _sample_result()
    result["verdicts"] = rubric.compute_verdicts(result["claims"], result["evidence"])
    out = to_perspective_findings(result, completion)
    gaps = out["market_findings"]["gaps"]

    def has(technology, criterion):
        return any(g["technology"] == technology and g["criterion"] == criterion and g["perspective"] == "market" for g in gaps)

    assert has("sw", "상용화·채택 현황") and has("sw", "반대·한계 근거") and has("hw", "반대·한계 근거")


def test_v08_claim은_technology_perspective_limitations를_채운다():
    result, completion = _sample_result()
    result["verdicts"] = rubric.compute_verdicts(result["claims"], result["evidence"])
    out = to_perspective_findings(result, completion)
    claim = out["market_findings"]["claims"][0]
    assert claim["technology"] == "sw" and claim["perspective"] == "market"
    assert claim["limitations"] == ["u"]  # 내부 Claim.uncertainty


def test_v08_claim_text는_240자로_자른다():
    result, completion = _sample_result()
    result["claims"][0]["statement"] = "가" * 300
    result["verdicts"] = rubric.compute_verdicts(result["claims"], result["evidence"])
    out = to_perspective_findings(result, completion)
    assert len(out["market_findings"]["claims"][0]["text"]) == 240


# --- 가짜 LLM: 전체 흐름 ------------------------------------------------------


class FakeStructured:
    def __init__(self, schema, resolve):
        self.schema, self.resolve = schema, resolve

    def invoke(self, messages):
        r = self.resolve(self.schema, messages)
        if isinstance(r, Exception):
            raise r
        return r

    def batch(self, batches, config=None, return_exceptions=False):
        out = []
        for messages in batches:
            try:
                out.append(self.resolve(self.schema, messages))
            except Exception as e:  # noqa: BLE001
                if not return_exceptions:
                    raise
                out.append(e)
        return out


class FakeLLM:
    def __init__(self, resolve):
        self.resolve = resolve

    def with_structured_output(self, schema):
        return FakeStructured(schema, self.resolve)


def _human_text(messages):
    return next(m for role, m in messages if role == "human")


def default_resolve(schema, messages):
    text = _human_text(messages)
    if schema is PlanOut:
        raise RuntimeError("템플릿 대체 경로 확인용: plan은 항상 실패시켜 폴백을 테스트한다")
    if schema is Judgement:
        m = re.search(r"본문: (.*)$", text, re.S)
        content = (m.group(1) if m else "").strip()
        is_counter = "반대·한계 근거" in text
        return Judgement(relevant=True, reason="fake", statement=content[:150], supporting_quote=content[:80],
                          evidence_level="announcement", stance="counter" if is_counter else "support",
                          scope="direct", source_type="other", organization="fake.org", published_date=None)
    if schema is CounterCheck:
        return CounterCheck(limited_subject="target", subject_is_target=True, is_counter=True, reason="fake")
    if schema is CompareOut:
        return CompareOut(items=[])
    raise AssertionError(f"unexpected schema {schema}")


def make_state():
    """AppState(graph/state.py) 모양의 최소 fixture."""
    return {
        "selected_tech": {
            "sw": {"name": SW, "short_name": "MLA", "technology": "sw", "approach": "attention",
                   "source_ids": [], "selection_reason": "MLA로 KV cache를 압축"},
            "hw": {"name": HW, "short_name": "CXL-Hybrid", "technology": "hw", "approach": "memory",
                   "source_ids": [], "selection_reason": "CXL-Hybrid로 메모리 계층 확장"},
        },
        "domain": "datacenter_inference",
        "request": {"as_of": AS_OF, "language": "ko", "scope": "datacenter_inference", "max_search_rounds": 2},
    }


def stub_web_search(query):
    # 출처 등급 필터(rag/tier.py)를 통과해야 흐름 검증이 가능하므로 등급표에 있는 도메인을 쓴다.
    return [{"title": f"기사: {query}", "url": f"https://www.reuters.com/{abs(hash(query))}",
             "content": f"{query}에 대한 데이터센터 관련 근거 문장입니다.", "organization": "Reuters",
             "published_date": "2026-07", "source_type": "news"}]


def test_전체_흐름은_8칸을_채우고_AppState_형태로_변환된다():
    deps = MarketAgentDeps(llm=FakeLLM(default_resolve), web_search=stub_web_search, retriever=None)
    out = make_node(deps)(make_state())
    findings = out["market_findings"]
    assert findings["perspective"] == "market"
    criteria = {(r["technology"], r["criterion"]) for r in findings["records"]}
    assert criteria == {(t, c) for t in ("sw", "hw") for c in ("시장 규모·성장성", "상용화·채택 현황", "생태계 지지")}
    assert out["evidence_store"]  # 근거가 실제로 쌓였다
    assert "market" in out["search_log_by_perspective"]


def test_인용_환각은_전부_기각된다():
    def resolve(schema, messages):
        if schema is Judgement:
            j = default_resolve(schema, messages)
            return j.model_copy(update={"supporting_quote": "본문에 없는 지어낸 인용"})
        return default_resolve(schema, messages)

    deps = MarketAgentDeps(llm=FakeLLM(resolve), web_search=stub_web_search, retriever=None)
    out = make_node(deps)(make_state())
    assert out["market_findings"]["claims"] == []
    assert out["market_findings"]["status"] == "partial"


def test_반대_근거_재검증이_주어_불일치를_걸러낸다():
    """ITME 초록처럼 '기존 방식의 한계'를 대상 기술 자신의 한계로 옮겨 적은 경우를 재현한다."""

    def resolve(schema, messages):
        if schema is CounterCheck:
            return CounterCheck(limited_subject="기존 오프로딩 방식", subject_is_target=False, is_counter=False, reason="fake")
        return default_resolve(schema, messages)

    deps = MarketAgentDeps(llm=FakeLLM(resolve), web_search=stub_web_search, retriever=None)
    out = make_node(deps)(make_state())
    assert not any(c["text"] for c in out["market_findings"]["claims"] if "반대" in c.get("text", ""))
    assert any(g["criterion"] == "반대·한계 근거" for g in out["market_findings"]["gaps"])


def test_RAG_어댑터는_채택_생태계_반대_칸에서만_호출된다():
    calls = []

    def spy_retriever(query, tech_id, k):
        calls.append((query, tech_id))
        return fake_retriever(query, tech_id, k)

    deps = MarketAgentDeps(llm=FakeLLM(default_resolve), web_search=lambda q: [], retriever=make_rag_fn(spy_retriever))
    make_node(deps)(make_state())
    assert calls and {tid for _, tid in calls} <= {"sw", "hw"}


def test_project_input은_다른_관점_결과를_건드리지_않는다():
    state = make_state()
    state["stakeholder_findings"] = {"positions": ["다른 관점 결론"]}
    projected = project_input(state)
    assert "stakeholder_findings" not in projected and "positions" not in str(projected)


def test_summarize_technical_은_없으면_빈_문자열():
    assert summarize_technical(None) == ""
    assert "MLA" in summarize_technical({"claims": [{"technology": "sw", "text": "MLA는 상용 서비스 중"}]})


if __name__ == "__main__":
    import inspect

    tests = [f for name, f in list(globals().items()) if name.startswith("test_") and inspect.isfunction(f)]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n전체 {len(tests)}개 통과")
