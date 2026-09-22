"""웹 검색 계층: 출처 정책, 캐시, 재시도·타임아웃·레이트리밋, 오프라인 목 모드.

검색 제공자의 도메인 필터는 신뢰하지 않는다. Tavily의 include_domains에 13개를 지정했더니
medium·substack·youtube가 그대로 반환되는 것을 확인했다(2개일 때는 동작).
따라서 출처 정책은 받은 뒤 이쪽에서 다시 적용하고, 거른 이유를 검색 로그에 남긴다.

캐시를 쓰는 이유는 재현성이다. 같은 질의가 실행 시점마다 다른 결과를 주면
ablation 비교도, 보고서 재생성도 성립하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from agents.domain.tools.evidence import SearchLogEntry, SourceType

# 출처 등급. 도메인 평가 근거는 공신력 순서가 판단 강도와 직결되므로 등급을 명시적으로 둔다.
SOURCE_TIERS: dict[SourceType, tuple[str, ...]] = {
    "paper": (
        "arxiv.org", "ieee.org", "acm.org", "dl.acm.org", "usenix.org",
        "openreview.net", "mlsys.org", "neurips.cc", "sciencedirect.com",
    ),
    "patent": ("patents.google.com", "patentscope.wipo.int", "uspto.gov"),
    "standard": ("computeexpresslink.org", "jedec.org", "opencompute.org", "snia.org"),
    "vendor": (
        "nvidia.com", "intel.com", "amd.com", "samsung.com", "skhynix.com",
        "micron.com", "marvell.com", "asteralabs.com", "rambus.com",
        "microsoft.com", "google.com", "meta.com", "deepseek.com",
    ),
    "news": (
        "reuters.com", "bloomberg.com", "theregister.com", "tomshardware.com",
        "semianalysis.com", "nextplatform.com", "hpcwire.com",
        "blocksandfiles.com", "servethehome.com", "anandtech.com",
    ),
}


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    source_type: SourceType
    published_date: str | None = None
    score: float = 0.0


# 벤더 도메인이라도 포럼·커뮤니티 서브도메인은 공식 문서가 아니다.
# 실측에서 forums.developer.nvidia.com 이 vendor 로 분류돼 근거 강도가 과대평가됐다.
COMMUNITY_SUBDOMAINS = ("forums.", "forum.", "community.", "discuss.", "answers.", "blogs.")


def _host(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def classify_source(url: str) -> SourceType | None:
    """등급표에 없는 도메인은 None. 근거로 채택하지 않는다."""
    host = _host(url)
    if host.startswith(COMMUNITY_SUBDOMAINS):
        return None  # 게시판 글은 근거로 채택하지 않는다
    for source_type, domains in SOURCE_TIERS.items():
        if any(host == d or host.endswith("." + d) for d in domains):
            return source_type
    return None


def allowed_domains() -> list[str]:
    return sorted({d for domains in SOURCE_TIERS.values() for d in domains})


class SearchCache:
    """질의 단위 파일 캐시. 같은 질의는 같은 결과를 돌려줘 실행 간 재현성을 유지한다."""

    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None  # 깨진 캐시는 없는 것으로 보고 다시 받는다

    def put(self, key: str, payload: dict) -> None:
        try:
            self._path(key).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass  # 캐시 기록 실패가 검색을 막아서는 안 된다


class TavilyWebSearch:
    """Tavily 검색. 출처 정책·캐시·재시도·레이트리밋을 이 계층에서 책임진다."""

    provider_name = "tavily"

    def __init__(
        self,
        cache: SearchCache,
        max_results: int = 6,
        timeout_seconds: float = 20.0,
        max_retries: int = 2,
        min_interval_seconds: float = 0.7,
    ):
        from langchain_tavily import TavilySearch

        self.max_results = max_results
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.min_interval = min_interval_seconds
        self._last_call = 0.0
        self._cache = cache
        self._tool = TavilySearch(
            max_results=max_results,
            search_depth="advanced",
            include_domains=allowed_domains(),
        )

    def _cache_key(self, query: str) -> str:
        return f"{self.provider_name}|{self.max_results}|advanced|{query}"

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()

    def _call_provider(self, query: str) -> tuple[list[dict], str | None]:
        """지수 백오프 재시도. 마지막 실패는 예외 대신 오류 문자열로 돌려준다."""
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                self._throttle()
                raw = self._tool.invoke({"query": query})
                return (raw.get("results", []) if isinstance(raw, dict) else []), None
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    time.sleep(1.5 * (2**attempt))
        return [], last_error

    def search(self, query: str) -> tuple[list[SearchResult], SearchLogEntry]:
        log = SearchLogEntry(
            query=query, provider=self.provider_name, requested_at=SearchLogEntry.now()
        )
        cached = self._cache.get(self._cache_key(query))
        if cached is not None:
            raw_results, log.from_cache = cached.get("results", []), True
        else:
            raw_results, error = self._call_provider(query)
            log.error = error
            if error is None:
                self._cache.put(self._cache_key(query), {"query": query, "results": raw_results})

        results: list[SearchResult] = []
        for item in raw_results:
            url = (item.get("url") or "").strip()
            snippet = (item.get("content") or "").strip()
            log.result_urls.append(url)
            if not url or not snippet:
                log.rejected.append({"url": url, "reason": "빈 URL 또는 본문"})
                continue
            source_type = classify_source(url)
            if source_type is None:
                # 제공자 필터가 흘려보낸 비신뢰 출처를 여기서 막는다.
                log.rejected.append({"url": url, "reason": "출처 등급표에 없는 도메인"})
                continue
            log.accepted_urls.append(url)
            results.append(
                SearchResult(
                    url=url,
                    title=item.get("title") or url,
                    snippet=snippet,
                    source_type=source_type,
                    published_date=item.get("published_date"),
                    score=float(item.get("score") or 0.0),
                )
            )
        return results, log


class OfflineMockSearch:
    """API 키·네트워크 없이 파이프라인을 돌리기 위한 목 제공자.

    테스트와 CI에서 쓰며, 캐시된 결과가 있으면 그것을 재생한다.
    """

    provider_name = "offline_mock"

    def __init__(self, cache: SearchCache, fixtures: dict[str, list[dict]] | None = None):
        self._cache = cache
        self._fixtures = fixtures or {}

    def search(self, query: str) -> tuple[list[SearchResult], SearchLogEntry]:
        log = SearchLogEntry(
            query=query, provider=self.provider_name, requested_at=SearchLogEntry.now(),
            from_cache=True,
        )
        raw = self._fixtures.get(query)
        if raw is None:
            cached = self._cache.get(f"tavily|6|advanced|{query}")
            raw = cached.get("results", []) if cached else []
        results = []
        for item in raw:
            url = item.get("url", "")
            source_type = classify_source(url)
            log.result_urls.append(url)
            if source_type is None:
                log.rejected.append({"url": url, "reason": "출처 등급표에 없는 도메인"})
                continue
            log.accepted_urls.append(url)
            results.append(
                SearchResult(
                    url=url, title=item.get("title") or url,
                    snippet=(item.get("content") or "").strip(),
                    source_type=source_type,
                    published_date=item.get("published_date"),
                    score=float(item.get("score") or 0.0),
                )
            )
        return results, log


def build_search_provider(cache_dir: Path, offline: bool | None = None):
    """offline 이 지정되지 않으면 TAVILY_API_KEY 유무로 결정한다."""
    cache = SearchCache(cache_dir)
    if offline is None:
        offline = not os.environ.get("TAVILY_API_KEY")
    return OfflineMockSearch(cache) if offline else TavilyWebSearch(cache)
