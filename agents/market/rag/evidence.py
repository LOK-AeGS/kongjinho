"""근거 저장소: 결정적 ID와 멱등 병합.

도메인 에이전트(agents/domain/rag/evidence.py)와 같은 설계를 시장 에이전트에 맞춰 다시 구현한다.
"다른 에이전트 폴더를 import하지 않는다"는 팀 규칙 때문에 공유하지 않고 각자 둔다.

재시도할 때마다 같은 근거가 다른 ID로 중복되면 evidence_store 병합이 깨진다.
그래서 ID는 출처 신원(정규화 URL + 위치 + 인용문)에서 해시로 유도해 같은 자료를
다시 가져와도 같은 ID가 나오게 한다.
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import urlsplit, urlunsplit

ID_PREFIX = "market:ev"
_WS = re.compile(r"\s+")
_TRACKING_PARAMS = ("utm_", "fbclid", "gclid", "ref=", "mc_cid", "mc_eid")


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    host = parts.hostname.lower() if parts.hostname else ""
    host = host[4:] if host.startswith("www.") else host
    query = "&".join(q for q in parts.query.split("&") if q and not any(t in q.lower() for t in _TRACKING_PARAMS))
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower() or "https", host, path, query, ""))


def normalize_text(text: str) -> str:
    return _WS.sub(" ", (text or "").replace("­", "")).strip()


def make_evidence_id(url: str, locator: str, quote: str) -> str:
    raw = f"{normalize_url(url)}|{locator}|{normalize_text(quote)}"
    return f"{ID_PREFIX}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"


def content_checksum(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
