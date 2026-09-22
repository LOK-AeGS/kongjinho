"""출처 등급 분류·필터 (설계서 §1.4 출처 우선순위).

Tavily 같은 검색 API가 반환한 도메인을 그대로 신뢰하지 않는다. 검색 결과를 받은 뒤
이 모듈로 다시 등급을 매기고, 등급표에 없는 도메인("기타")은 채택하지 않는다
(이유는 search_log에 남긴다).
"""

from __future__ import annotations

from urllib.parse import urlparse

Tier = str  # "paper" | "patent" | "standard" | "vendor" | "research" | "news" | "other"

# 2026-09-22 실 API 실행 3회(search_log)에서 실제로 걸러진 도메인 중 §1.4상 정당한 출처를
# 재검토해 추가했다: 대상 기술(DeepSeek) 원저작사, 서빙 프레임워크 관련 주요 벤더,
# 정식 시장조사기관, arxiv 미러. 그 외 개인 블로그·SNS·포럼은 그대로 "기타"로 남긴다.
_PAPER_DOMAINS = (
    "arxiv.org", "aclanthology.org", "openreview.net", "dl.acm.org", "ieee.org", "usenix.org",
    "alphaxiv.org", "ar5iv.org", "papers.cool", "semanticscholar.org",
)
_PATENT_DOMAINS = ("patents.google.com", "patft.uspto.gov", "worldwide.espacenet.com", "wipo.int")
_STANDARD_DOMAINS = ("jedec.org", "opencompute.org", "computeexpresslink.org", "opencapi.org", "snia.org")
_VENDOR_DOMAINS = (
    "nvidia.com", "amd.com", "intel.com", "samsung.com", "skhynix.com", "sk.com",
    "microsoft.com", "azure.com", "aws.amazon.com", "cloud.google.com", "huggingface.co",
    "github.com", "openai.com", "anthropic.com", "meta.com", "micron.com",
    "deepseek.com",  # 대상 SW 기술(DeepSeek-V2)의 원저작사 공식 도메인
    "databricks.com", "redhat.com", "vllm.ai",  # 서빙 프레임워크·추론 인프라 주요 벤더
)
# 정식 시장조사기관(§1.4 "2차 출처: 학술산업 분석"). 뉴스와 분리해 관리한다 — 근거 강도를
# 구분해야 할 때(예: 1차 발표 vs 3자 분석) 뒤에 쓸 수 있게.
_RESEARCH_DOMAINS = ("mordorintelligence.com", "idc.com", "gartner.com", "counterpointresearch.com", "trendforce.com")
_NEWS_DOMAINS = (
    "reuters.com", "theregister.com", "semianalysis.com", "techcrunch.com", "bloomberg.com",
    "wsj.com", "zdnet.com", "theverge.com", "arstechnica.com", "yonhapnews.co.kr", "etnews.com",
)
# 등급 우선순위: 이 목록에 걸리면 신뢰 근거. 없으면 "other"(기타) — 채택하지 않는다.
_TIER_TABLE: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("paper", _PAPER_DOMAINS),
    ("patent", _PATENT_DOMAINS),
    ("standard", _STANDARD_DOMAINS),
    ("vendor", _VENDOR_DOMAINS),
    ("research", _RESEARCH_DOMAINS),
    ("news", _NEWS_DOMAINS),
)


def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def classify(url: str) -> Tier:
    host = domain_of(url)
    for tier, domains in _TIER_TABLE:
        if any(host == d or host.endswith(f".{d}") for d in domains):
            return tier
    return "other"


def is_trusted(url: str) -> bool:
    """§1.4: '그 외(등급표에 없는 도메인)'는 채택하지 않는다."""
    return classify(url) != "other"


def trusted_domains() -> list[str]:
    """등급표의 모든 도메인. 검색 단계에서 후보를 좁히는 데 쓴다(Tavily include_domains)."""
    return [domain for _, domains in _TIER_TABLE for domain in domains]


def filter_trusted(results: list[dict]) -> tuple[list[dict], list[str]]:
    """신뢰 등급 결과만 남긴다. (통과한 결과, 걸러진 사유 로그)를 돌려준다."""
    kept: list[dict] = []
    dropped: list[str] = []
    for r in results:
        if is_trusted(r["url"]):
            kept.append(r)
        else:
            dropped.append(f"출처 등급 필터: '{domain_of(r['url'])}'는 등급표에 없어 채택하지 않음 ({r['url']})")
    return kept, dropped
