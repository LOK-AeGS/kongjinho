"""기술 조사 주장-인용 수치 결합 회귀 테스트.

고정 Pool A 논문(SHA-256 매니페스트)을 정답지로 쓴다. 정답지는 테스트에서만 쓰고 런타임에 주입하지 않는다.
API 키 없이 실행된다(PDF 파싱만 사용).
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agents.technical.corpus import default_source_dir, load_manifest, normalize_text, parse_corpus  # noqa: E402
from agents.technical.subgraph import _bind_numbers  # noqa: E402
from graph.metrics import unsupported_values  # noqa: E402

FACTS = json.loads((Path(__file__).with_name("fixtures") / "key_facts.json").read_text(encoding="utf-8"))["facts"]


@lru_cache(maxsize=1)
def chunks():
    return [chunk.to_dict() for chunk in parse_corpus(load_manifest(), default_source_dir())]


def candidates(doc_id, pages):
    return {
        chunk["chunk_id"]: chunk
        for chunk in chunks()
        if chunk["doc_id"] == doc_id and chunk["page"] in pages
    }


def bound_to(chunk, quote):
    assert quote in normalize_text(chunk["text"])
    return [{"candidate_id": chunk["chunk_id"], "doc_id": chunk["doc_id"], "quote": quote}]


def test_key_fact_sentences_exist_on_their_pages():
    """정답지 문장이 고정 PDF의 해당 쪽에 실제로 있다(코퍼스·파서가 바뀌면 깨진다)."""
    for fact in FACTS:
        page_text = " ".join(
            normalize_text(chunk["text"])
            for chunk in chunks()
            if chunk["doc_id"] == fact["doc_id"] and chunk["page"] == fact["page"]
        )
        assert normalize_text(fact["sentence"]) in page_text, fact


def test_number_matching_ignores_unit_and_latex_notation():
    # live 1~5차 오탐 원인: 원문 '5.76 times', '35.7\%'를 단위가 붙은 형태로만 찾았다.
    assert unsupported_values({"5.76"}, "which is 5.76 times the maximum") == set()
    assert unsupported_values({"35.7"}, r"up to a 35.7\% throughput") == set()
    assert unsupported_values({"1.80"}, "a 1.80× throughput improvement") == set()
    assert unsupported_values({"1.80"}, "a 11.80× value and 1.805") == {"1.80"}


def test_misplaced_citation_is_relocated_to_page_with_the_number():
    # live 실행: 'NVMe-oF 대비 1.80배'를 10쪽으로 인용했지만 수치는 2쪽에 있다.
    doc = "arxiv:2606.12556v2"
    pool = candidates(doc, {2, 9, 10})
    cited = next(chunk for chunk in pool.values() if chunk["page"] == 10)
    quote = normalize_text(cited["text"])[:120]
    added, violations = _bind_numbers(
        "ITME는 NVMe-oF 대비 최대 1.80배 처리량 향상", bound_to(cited, quote), pool, "technical:claim:t", "unknown",
    )
    assert violations == []
    evidence = next(iter(added.values()))
    assert evidence["page_or_locator"].startswith("p.2:")
    assert "1.80" in evidence["quote"] and "NVMe-oF" in evidence["quote"]


def test_number_in_english_unit_is_bound_from_cited_chunk():
    doc = "arxiv:2405.04434v5"
    pool = candidates(doc, {16})
    cited = next(chunk for chunk in pool.values() if "5.76" in chunk["text"])
    quote = normalize_text(cited["text"])[:80]
    assert "5.76" not in quote
    added, violations = _bind_numbers(
        "8×H800 단일 노드에서 생성 처리량이 DeepSeek 67B의 5.76배", bound_to(cited, quote), pool, "technical:claim:t", "unknown",
    )
    assert violations == []
    assert any("5.76 times" in evidence["quote"] for evidence in added.values())


def test_same_number_in_unrelated_context_is_not_bound():
    # DeepSeek-V2 41쪽 "35.7% female"을 MLA 성능 근거로 붙이면 안 된다.
    doc = "arxiv:2405.04434v5"
    pool = candidates(doc, {4, 41})
    cited = next(chunk for chunk in pool.values() if chunk["page"] == 4)
    assert any("35.7" in chunk["text"] for chunk in pool.values() if chunk["page"] == 41)
    quote = normalize_text(cited["text"])[:100]
    added, violations = _bind_numbers(
        "MLA는 처리량을 35.7% 높였다", bound_to(cited, quote), pool, "technical:claim:t", "unknown",
    )
    assert added == {}
    assert violations and violations[0].startswith("인용문에 없는 수치 35.7")
