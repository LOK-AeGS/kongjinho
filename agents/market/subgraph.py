"""시장 평가 에이전트 내부 그래프.

plan → search → organize → check ─(부족한 칸 있고 재시도 여유 있음)→ search로 되돌아감
                              └(충족 또는 한도 도달)→ compare → finalize

- plan: LLM이 기술×관점(size/adoption/ecosystem/counter)별 검색 질의 생성
- search: 웹 검색 + (채택·생태계·반대 근거는) RAG 검색
- organize: LLM이 관련성·범위·stance·증거 수준을 판단하고, 코드가 인용 원문을 검증
  → 반대 근거는 별도 LLM 호출로 한 번 더 검증한다(한계의 주어가 대상 기술인지 확인)
- check: rubric.py가 기술×기준별 판정을 계산해 부족한 칸만 다른 질의로 재검색
- compare: LLM이 채택 동인·도입 비용·확산 장벽을 inference 주장으로 추론
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable

from langgraph.graph import END, START, StateGraph

from agents.market.prompts import (
    ASPECT_DEFS,
    ASPECT_LABELS,
    COMPARE_SYSTEM,
    COUNTER_CHECK_SYSTEM,
    JUDGE_SYSTEM,
    PLAN_SYSTEM,
    PROMPT_VERSION,
    TEMPLATE_QUERIES,
    CompareOut,
    CounterCheck,
    Judgement,
    PlanOut,
)
from agents.market.quality import linter, rubric
from agents.market.rag import index as body_index
from agents.market.rag import tier
from agents.market.rag.fetch import FetchedDocument, PAGE_BUDGET
from agents.market.state import Claim, Completion, InternalEvidence, MarketLocal, new_evidence

TECH_IDS = ("sw", "hw")
RAG_ASPECTS = {"adoption", "ecosystem", "counter"}
_DATE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")


@dataclass
class MarketAgentDeps:
    """State 밖 런타임 의존성. JSON 직렬화 대상이 아니다."""

    llm: object  # 계획·판정용 (호출 빈도가 높음). .with_structured_output(...) 을 지원해야 한다.
    strong_llm: object | None = None  # 반대 근거 재검증·비교 추론용. None이면 llm을 그대로 쓴다.
    web_search: Callable[[str], list[dict]] = None  # (query) -> list[SearchResult]
    retriever: Callable[[str, str | None], list[dict]] | None = None  # (query, tech_id) -> list[SearchResult]. None이면 RAG 건너뜀.
    # Pool B(런타임 수집 코퍼스): 채택·생태계·반대 칸의 웹 검색 결과 본문을 가져온다.
    # None이면 기존처럼 검색 스니펫만 쓴다(오프라인 테스트 기본값).
    fetch_body: Callable[[str], FetchedDocument] | None = None
    page_budget: int = PAGE_BUDGET  # 본문 수집 누적 한도(agents/market/rag/fetch.py 기준 페이지 환산)

    def strong(self):
        return self.strong_llm or self.llm


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _all_slots() -> list[str]:
    return [f"{t}:{a}" for t in TECH_IDS for a in ASPECT_LABELS]


def _rag_query(queries: list[str]) -> str:
    """논문은 영문이라 RAG에는 영문 비율이 높은 질의를 우선 쓴다."""
    for q in reversed(queries):
        if sum(ch.isascii() for ch in q) / max(len(q), 1) > 0.9:
            return q
    return queries[-1]


def _plan_template(techs: dict[str, dict], domain: str, year: str) -> dict[str, list[str]]:
    bank: dict[str, list[str]] = {}
    for tid, t in techs.items():
        for ak, templates in TEMPLATE_QUERIES.items():
            bank[f"{tid}:{ak}"] = [tpl.format(tech=t["name"], domain=domain, year=year) for tpl in templates]
    return bank


def plan(s: MarketLocal, deps: MarketAgentDeps) -> dict:
    techs = {t: {"name": s["technologies"][t], "desc": s["tech_desc"][t]} for t in TECH_IDS}
    year = s["as_of_date"][:4]
    errors: list[str] = []
    bank: dict[str, list[str]] = {}
    try:
        desc = "\n".join(f"- {tid}: {t['name']} ({t['desc']})" for tid, t in techs.items())
        human = f"대상 기술:\n{desc}\n도메인: {s['domain']}\n기준 연도: {year}"
        if s.get("technical_summary"):
            human += f"\n\n선행 기술 조사 요약:\n{s['technical_summary']}"
        out: PlanOut = deps.llm.with_structured_output(PlanOut).invoke([("system", PLAN_SYSTEM), ("human", human)])
        bank = {f"{i.technology_id}:{i.aspect}": [q for q in i.queries if q.strip()] for i in out.items}
    except Exception as e:
        errors.append(f"질의 계획 실패, 템플릿으로 대체: {type(e).__name__}: {e}")
    fallback = _plan_template(techs, s["domain"], year)
    for sk in _all_slots():
        if not bank.get(sk):
            bank[sk] = fallback[sk]
    pending = [{"slot_key": sk, "text": bank[sk][0]} for sk in _all_slots()]
    return {
        "query_bank": bank, "pending": pending, "search_rounds_used": 0, "seen": [], "quote_keys": [],
        "raw": [], "evidence": [], "claims": [], "adoption_cases": [], "verdicts": [], "errors": errors,
        "next_action": "search", "pages_used": 0, "search_log": [],
    }


def _collect_body(
    results: list[dict], query: str, deps: MarketAgentDeps, pages_used: int
) -> tuple[list[dict], int, list[dict]]:
    """Pool B: 출처 등급 필터 -> 본문 수집(예산 검사) -> 길이 판단 -> 짧음: 그대로 / 김: 청킹+색인 후 질의 매칭 조각.

    스니펫만 남은 결과(본문을 못 가져온 것)는 채택 판단 근거로 약해 버린다.
    """
    trusted, dropped = tier.filter_trusted(results)
    log = [{"message": m} for m in dropped]
    out: list[dict] = []
    for r in trusted:
        if pages_used >= deps.page_budget:
            log.append({"message": f"{deps.page_budget}p 한도 도달로 본문 수집 중단"})
            break
        doc = deps.fetch_body(r["url"])
        if doc.error:
            log.append({"message": f"본문 수집 실패 '{r['url']}': {doc.error}"})
            continue
        if pages_used + doc.page_count > deps.page_budget:
            log.append({"message": f"{deps.page_budget}p 한도 초과로 제외({r['url']}, {doc.page_count}p)"})
            continue
        pages_used += doc.page_count
        if doc.is_long:
            chunks = body_index.chunk_document(doc)
            for c in body_index.search_chunks(chunks, query, k=3):
                out.append({**r, "content": c.text, "title": doc.title, "page": None, "section": c.locator})
        else:
            out.append({**r, "content": doc.full_text, "title": doc.title})
    return out, pages_used, log


def search(s: MarketLocal, deps: MarketAgentDeps) -> dict:
    raw: list[dict] = []
    errors = list(s["errors"])
    pages_used = s.get("pages_used", 0)
    search_log = list(s.get("search_log", []))
    for q in s["pending"]:
        tid, aspect = q["slot_key"].split(":")
        try:
            results = list(deps.web_search(q["text"]))
            # 출처 등급 필터는 관점과 무관하게 항상 적용한다. 예전에는 _collect_body 안에만
            # 있어서 RAG_ASPECTS 에 없는 size 질의의 결과가 걸러지지 않고 근거가 됐다.
            # 오프라인 배선 확인용 [STUB] 결과는 is_stub 로 따로 표시되므로 등급 판정에서 뺀다.
            stubs = [r for r in results if r["title"].startswith("[STUB]")]
            results, dropped = tier.filter_trusted(
                [r for r in results if not r["title"].startswith("[STUB]")]
            )
            results += stubs
            search_log += [{"message": m} for m in dropped]
            if aspect in RAG_ASPECTS:
                if deps.retriever is not None:
                    results += list(deps.retriever(_rag_query(s["query_bank"][q["slot_key"]]), tid))
                if deps.fetch_body is not None:
                    results, pages_used, log = _collect_body(results, q["text"], deps, pages_used)
                    search_log += log
            raw += [{"slot_key": q["slot_key"], "result": r} for r in results]
        except Exception as e:
            errors.append(f"검색 실패 '{q['text']}': {type(e).__name__}: {e}")
    return {
        "raw": raw, "pending": [], "errors": errors, "search_rounds_used": s["search_rounds_used"] + 1,
        "pages_used": pages_used, "search_log": search_log,
    }


def _uncertainty(level: str | None, source_type: str, scope: str, is_stub: bool) -> str:
    notes = ["공개 정보 기반 추정"]
    if is_stub:
        notes.append("[STUB] 실제 근거가 아님")
    if level == "forecast":
        notes.append("전망치이며 실측이 아님")
    if source_type == "community":
        notes.append("비공식 출처(블로그·커뮤니티)")
    if source_type == "paper":
        notes.append("논문 자체 보고이며 실운용 검증이 아님")
    if scope == "class":
        notes.append("대상 기술이 아닌 기술군 수준의 근거")
    return "; ".join(notes)


def organize(s: MarketLocal, deps: MarketAgentDeps) -> dict:
    evidence = list(s["evidence"])
    claims = list(s["claims"])
    seen = set(s["seen"])
    quote_keys = set(s["quote_keys"])
    errors = list(s["errors"])
    today = date.today().isoformat()
    accessed_at = datetime.now(timezone.utc).isoformat()

    todo = []
    for item in s["raw"]:
        r = item["result"]
        key = f"{item['slot_key']}|{r['url']}|{r.get('page')}"
        if key in seen:
            continue
        tid, aspect = item["slot_key"].split(":")
        tech_name, tech_desc = s["technologies"][tid], s["tech_desc"][tid]
        user = (
            f"대상 기술: {tech_name} ({tech_desc})\n도메인: {s['domain']}\n조사 관점: {ASPECT_LABELS[aspect]}\n"
            f"조사 관점 정의: {ASPECT_DEFS[aspect]}\n\n[검색 결과]\n제목: {r['title']}\nURL: {r['url']}\n본문: {r['content']}"
        )
        todo.append((key, item, [("system", JUDGE_SYSTEM), ("human", user)]))

    judgements: list = []
    if todo:
        try:
            judgements = deps.llm.with_structured_output(Judgement).batch(
                [t[2] for t in todo], config={"max_concurrency": 6}, return_exceptions=True
            )
        except Exception as e:
            errors.append(f"판단 일괄 실패: {type(e).__name__}: {e}")
            judgements = [e] * len(todo)

    rejected = 0
    accepted: list[tuple[dict, Judgement]] = []
    for (key, item, _), j in zip(todo, judgements):
        if isinstance(j, Exception):
            errors.append(f"판단 실패 {item['result']['url']}: {type(j).__name__}")
            continue  # seen에 넣지 않아 다음 라운드에 다시 시도할 수 있다.
        seen.add(key)
        r = item["result"]
        aspect = item["slot_key"].split(":")[1]
        quote_ok = bool(_norm(j.supporting_quote)) and _norm(j.supporting_quote) in _norm(r["content"])
        if not j.relevant or not j.statement.strip() or not quote_ok:
            rejected += 1
            continue
        if aspect == "counter" and j.stance != "counter":
            rejected += 1
            continue
        accepted.append((item, j))

    # 반대 근거는 별도로 한 번 더 검증한다: 인용문의 한계가 진짜 대상 기술 자신의 것인지.
    need = [n for n, (_, j) in enumerate(accepted) if j.stance == "counter"]
    checks: list = []
    if need:
        try:
            batches = []
            for n in need:
                item, j = accepted[n]
                tid = item["slot_key"].split(":")[0]
                batches.append([
                    ("system", COUNTER_CHECK_SYSTEM),
                    ("human", f"대상 기술: {s['technologies'][tid]} ({s['tech_desc'][tid]})\n인용문: {j.supporting_quote}"),
                ])
            checks = deps.strong().with_structured_output(CounterCheck).batch(
                batches, config={"max_concurrency": 6}, return_exceptions=True
            )
        except Exception as e:
            checks = [e] * len(need)
    drop: set[int] = set()
    for n, res in zip(need, checks):
        if isinstance(res, Exception):
            errors.append(f"반대 근거 검증 실패 {accepted[n][0]['result']['url']}: {type(res).__name__}")
            drop.add(n)  # 검증 못 한 counter는 보수적으로 버린다.
        elif not res.is_counter:
            drop.add(n)

    for n, (item, j) in enumerate(accepted):
        if n in drop:
            rejected += 1
            continue
        r = item["result"]
        tid, aspect = item["slot_key"].split(":")
        qk = f"{tid}|{r['url']}|{r.get('page')}|{_norm(j.supporting_quote)[:80]}"
        if qk in quote_keys:
            rejected += 1  # 같은 근거·같은 인용이 다른 칸에 이미 쓰였다.
            continue
        quote_keys.add(qk)

        is_paper = bool(r.get("document_id")) or j.source_type == "paper"
        level = "pilot" if (is_paper and j.evidence_level == "production") else j.evidence_level
        published = j.published_date if j.published_date and _DATE_RE.match(j.published_date) else r.get("published_date")
        ev: InternalEvidence = new_evidence(
            url=r["url"], locator=(f"p.{r['page']}" if r.get("page") else r["url"]), title=r["title"],
            organization=j.organization or r.get("organization", ""), source_type="paper" if is_paper else j.source_type,
            published_at=published, accessed_at=today, quote=j.supporting_quote,
            doc_id=r.get("document_id"), scope=j.scope,
        )
        existing = next((e for e in evidence if e["id"] == ev["id"]), None)
        if existing is None:
            evidence.append(ev)
        claim: Claim = {
            "claim_id": f"market:claim:{len(claims) + 1:03d}", "technology_id": tid, "aspect": aspect,
            "basis": "direct_evidence", "evidence_level": level, "statement": j.statement.strip(),
            "evidence_ids": [ev["id"]], "stance": j.stance, "scope": j.scope, "observed_at": published,
            "uncertainty": _uncertainty(level, "paper" if is_paper else j.source_type, j.scope, r["title"].startswith("[STUB]")),
            "is_stub": r["title"].startswith("[STUB]"),
        }
        claims.append(claim)

    return {
        "evidence": evidence, "claims": claims, "seen": sorted(seen), "quote_keys": sorted(quote_keys),
        "errors": errors, "raw": [],
    }


def check(s: MarketLocal) -> dict:
    missing = rubric.missing_slots(s["claims"], s["evidence"])
    if not missing:
        return {"next_action": "compare"}
    if s["search_rounds_used"] < s["max_search_rounds"]:
        attempt = s["search_rounds_used"]
        pending = [
            {"slot_key": sk, "text": s["query_bank"][sk][attempt % len(s["query_bank"][sk])]}
            for sk in missing
        ]
        return {"pending": pending, "next_action": "search"}
    return {"next_action": "compare"}


def compare(s: MarketLocal, deps: MarketAgentDeps) -> dict:
    direct = [c for c in s["claims"] if c["basis"] == "direct_evidence"]
    if not direct:
        return {"next_action": "finish"}
    payload = [
        {"claim_id": c["claim_id"], "tech": c["technology_id"], "topic": ASPECT_LABELS.get(c["aspect"], c["aspect"]),
         "stance": c["stance"], "statement": c["statement"]}
        for c in direct
    ]
    by_id = {c["claim_id"]: c for c in direct}
    claims = list(s["claims"])
    errors = list(s["errors"])
    try:
        lines = "\n".join(f"- {p['claim_id']} | {p['tech']} | {p['topic']} | stance={p['stance']} | {p['statement']}" for p in payload)
        out: CompareOut = deps.strong().with_structured_output(CompareOut).invoke(
            [("system", COMPARE_SYSTEM), ("human", f"도메인: {s['domain']}\n주장 목록:\n{lines}")]
        )
    except Exception as e:
        errors.append(f"시장 관점 비교 실패: {type(e).__name__}: {e}")
        return {"errors": errors, "next_action": "finish"}

    for item in out.items:
        if item.insufficient or not item.statement.strip():
            continue
        based = [cid for cid in dict.fromkeys(item.based_on_claim_ids) if cid in by_id and by_id[cid]["technology_id"] == item.technology_id]
        if not based:
            continue
        ev_ids = list(dict.fromkeys(e for cid in based for e in by_id[cid]["evidence_ids"]))
        claims.append({
            "claim_id": f"market:claim:{len(claims) + 1:03d}", "technology_id": item.technology_id,
            "aspect": f"{item.topic}(추론)", "basis": "inference", "evidence_level": None,
            "statement": item.statement.strip(), "evidence_ids": ev_ids, "stance": "neutral", "scope": "direct",
            "observed_at": None, "uncertainty": item.uncertainty.strip() or "수집된 공개 정보에 근거한 추론이며 검증되지 않음",
            "is_stub": False,
        })
    return {"claims": claims, "errors": errors, "next_action": "finish"}


def finalize(s: MarketLocal) -> dict:
    claims, evidence, names = s["claims"], s["evidence"], s["technologies"]
    verdicts = rubric.compute_verdicts(claims, evidence)
    gaps = rubric.gap_messages(claims, evidence, names)
    note = rubric.imbalance_note(claims, names)
    if note:
        gaps.append(note)
    lint_hits = linter.lint_statements([(c["claim_id"], c["statement"]) for c in claims])
    errors = list(s["errors"]) + [f"우열 어휘 '{h['matched']}' ({h['claim_id']})" for h in lint_hits]
    if not claims and not s["seen"]:
        # 근거가 하나도 판정되지 못했다(예: 검색·판단 자체가 전부 실패). plan()이 템플릿으로
        # 정상 폴백했거나 개별 근거가 품질 기준으로 거절된 것은 "failed"가 아니라 "partial"이다.
        status = "failed"
    elif gaps:
        status = "partial"
    else:
        status = "complete"
    completion: Completion = {
        "status": status, "search_rounds_used": s["search_rounds_used"], "revision_rounds_used": 0,
        "gaps": gaps, "errors": errors,
    }
    return {"claims": claims, "evidence": evidence, "verdicts": verdicts, "completion": completion, "prompt_version": PROMPT_VERSION}


def _route(s: MarketLocal) -> str:
    return s["next_action"] if s["next_action"] in ("search", "compare") else "compare"


def _build_subgraph(deps: MarketAgentDeps):
    g = StateGraph(MarketLocal)
    g.add_node("plan", lambda s: plan(s, deps))
    g.add_node("search", lambda s: search(s, deps))
    g.add_node("organize", lambda s: organize(s, deps))
    g.add_node("check", check)
    g.add_node("compare", lambda s: compare(s, deps))
    g.add_node("finalize", finalize)
    g.add_edge(START, "plan")
    g.add_edge("plan", "search")
    g.add_edge("search", "organize")
    g.add_edge("organize", "check")
    g.add_conditional_edges("check", _route, {"search": "search", "compare": "compare"})
    g.add_edge("compare", "finalize")
    g.add_edge("finalize", END)
    return g.compile()
