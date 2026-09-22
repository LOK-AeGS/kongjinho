"""Tavily Search + Extract 기반 데이터센터 운용 근거 수집."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .corpus import normalize_text


WEB_QUERY_TEMPLATES = {
    "sw": (
        '"DeepSeek-V2" MLA data center LLM inference serving pilot production deployment customer',
        '"DeepSeek-V2" MLA operational environment qualified system sustained production serving',
    ),
    "hw": (
        '"ITME" "CXL-Hybrid" data center LLM inference pilot deployment customer',
        '"Inference Tiered Memory Expansion" operational environment qualified production serving',
    ),
}

TECH_TERMS = {
    "sw": ("deepseek-v2", "deepseek v2", "multi-head latent attention", "mla"),
    "hw": ("itme", "inference tiered memory expansion", "cxl-hybrid"),
}
DATACENTER_TERMS = (
    "data center", "datacenter", "llm inference", "inference serving", "serving system",
    "gpu server", "production", "pilot", "deployment", "customer",
)
PRIMARY_DOMAINS = (
    "deepseek.com", "api-docs.deepseek.com", "github.com", "skhynix.com",
    "news.skhynix.com", "arxiv.org", "vllm.ai",
)


def web_query(technology: str, attempt: int) -> str:
    values = WEB_QUERY_TEMPLATES[technology]
    return values[min(max(attempt - 1, 0), len(values) - 1)]


def _is_relevant(text: str, technology: str) -> bool:
    blob = text.lower()
    technology_match = any(
        re.search(rf"\b{re.escape(term)}\b", blob) if term in {"mla", "itme"} else term in blob
        for term in TECH_TERMS[technology]
    )
    return technology_match and any(
        term in blob for term in DATACENTER_TERMS
    )


def _source_grade(url: str) -> tuple[str, str]:
    host = (urlsplit(url).hostname or "").lower()
    primary = any(host == domain or host.endswith("." + domain) for domain in PRIMARY_DOMAINS)
    return ("primary", "direct") if primary else ("secondary", "proxy")


class TavilyProvider:
    def __init__(self, client=None, cache_dir: Path | str | None = None):
        self.client = client
        self.cache_dir = Path(cache_dir or "data/technical/web_cache")

    def _client(self):
        if self.client is None:
            try:
                from tavily import TavilyClient
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Tavily 검색에는 tavily-python 패키지가 필요합니다.") from exc
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                raise RuntimeError("TAVILY_API_KEY가 설정되지 않았습니다.")
            self.client = TavilyClient(api_key=api_key)
        return self.client

    def search_and_extract(self, technology: str, query: str, as_of: str) -> tuple[list[dict], dict]:
        requested_at = datetime.now(timezone.utc).isoformat()
        search = self._client().search(
            query=query,
            search_depth="advanced",
            chunks_per_source=3,
            max_results=5,
            topic="general",
            include_answer=False,
            include_raw_content=False,
            include_published_date=True,
            end_date=as_of,
        )
        results = search.get("results", [])
        urls = [item["url"] for item in results if item.get("url")]
        log = {
            "provider": "tavily",
            "technology": technology,
            "query": query,
            "requested_at": requested_at,
            "search_request_id": search.get("request_id"),
            "search_urls": urls,
            "accepted_urls": [],
            "rejected": [],
            "extract_failed_urls": [],
            "status": "no_results" if not urls else "ok",
        }
        if not urls:
            return [], log

        extracted = self._client().extract(
            urls=urls,
            query=query,
            chunks_per_source=3,
            extract_depth="advanced",
            format="markdown",
            include_images=False,
        )
        log["extract_request_id"] = extracted.get("request_id")
        log["extract_failed_urls"] = [item.get("url") for item in extracted.get("failed_results", [])]
        metadata = {item["url"]: item for item in results if item.get("url")}
        candidates: list[dict] = []
        for item in extracted.get("results", []):
            url = item.get("url", "")
            raw = normalize_text(item.get("raw_content", ""))
            if not raw or not _is_relevant(raw, technology):
                log["rejected"].append({"url": url, "reason": "technology_or_datacenter_scope"})
                continue
            content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            primary_or_secondary, direct_or_proxy = _source_grade(url)
            meta = metadata.get(url, {})
            published_at = meta.get("published_date")
            if published_at:
                try:
                    if date.fromisoformat(published_at[:10]) > date.fromisoformat(as_of):
                        log["rejected"].append({"url": url, "reason": "after_as_of_date"})
                        continue
                except ValueError:
                    published_at = None
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            snapshot = self.cache_dir / f"{content_hash}.json"
            if not snapshot.exists():
                snapshot.write_text(
                    json.dumps({"url": url, "content": raw, "content_hash": content_hash}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            candidate_id = "technical:web:" + hashlib.sha256(
                f"{url}|{content_hash}".encode("utf-8")
            ).hexdigest()[:16]
            candidates.append(
                {
                    "chunk_id": candidate_id,
                    "doc_id": None,
                    "title": meta.get("title") or url,
                    "author_or_org": urlsplit(url).hostname or url,
                    "url": url,
                    "published_at": published_at,
                    "role": "web_operational",
                    "technology": technology,
                    "approach": "operational_evidence",
                    "page": None,
                    "locator": f"tavily-extract:{content_hash[:12]}",
                    "text": raw[:8000],
                    "content_hash": content_hash,
                    "origin": "tavily",
                    "source_type": "official_web" if primary_or_secondary == "primary" else "other",
                    "primary_or_secondary": primary_or_secondary,
                    "direct_or_proxy": direct_or_proxy,
                    "rrf_score": float(meta.get("score") or 0),
                    "queries": [query],
                    "matched_criteria": ["operational_evidence"],
                }
            )
            log["accepted_urls"].append(url)
        if urls and not candidates:
            log["status"] = "access_incomplete" if log["extract_failed_urls"] else "not_found"
        return candidates, log
