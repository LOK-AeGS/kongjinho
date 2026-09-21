"""원문 HTML snapshot과 locator. 검색 snippet을 근거로 사용하지 않는다."""
import hashlib
import ipaddress
import json
import re
import socket
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def normalize(text):
    return ' '.join(text.split())


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.metadata = [], {}
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ('script', 'style', 'noscript'):
            self.hidden += 1
        if tag == 'meta':
            key = attrs.get('property') or attrs.get('name')
            if key:
                self.metadata[key.lower()] = attrs.get('content', '')
        if tag in ('p', 'div', 'h1', 'h2', 'h3', 'li', 'tr', 'br', 'section'):
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'noscript'):
            self.hidden = max(0, self.hidden - 1)
        if tag in ('p', 'div', 'h1', 'h2', 'h3', 'li', 'tr', 'section'):
            self.parts.append('\n')

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def parse_html(html):
    parser = PageParser()
    parser.feed(html)
    blocks = [normalize(x) for x in ''.join(parser.parts).splitlines() if normalize(x)]
    return [{'locator': f'block:{i:04d}', 'text': text} for i, text in enumerate(blocks, 1)], parser.metadata


class PageFetcher:
    def __init__(self, cache_dir, *, client=None, min_interval=1.0, allowed_domains=()):
        self.cache_dir = Path(cache_dir)
        self.client = client or httpx.Client(timeout=20, follow_redirects=False)
        self.min_interval = min_interval
        self.allowed_domains = tuple(allowed_domains)
        self.last_request = 0.0
        self.memory = {}

    def _safe_url(self, url):
        p = urlparse(url)
        if p.scheme not in ('https', 'http') or not p.hostname or p.username or p.password:
            raise ValueError('invalid_url')
        if self.allowed_domains and not any(p.hostname == d or p.hostname.endswith('.' + d) for d in self.allowed_domains):
            raise ValueError('domain_filtered')
        for entry in socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == 'https' else 80)):
            if not ipaddress.ip_address(entry[4][0]).is_global:
                raise ValueError('non_public_address')

    def _get(self, url):
        for _ in range(4):
            self._safe_url(url)
            time.sleep(max(0, self.min_interval - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            with self.client.stream('GET', url, headers={'User-Agent': 'RAGStakeholderResearch/0.3'}) as r:
                if r.is_redirect:
                    target = urljoin(url, r.headers['location'])
                    if urlparse(target).netloc != urlparse(url).netloc:
                        raise ValueError('cross_origin_redirect_requires_review')
                    url = target
                    continue
                data = bytearray()
                for chunk in r.iter_bytes():
                    data.extend(chunk)
                    if len(data) > 2_000_000:
                        raise ValueError('response_too_large')
                return r.status_code, dict(r.headers), bytes(data), str(r.url)
        raise ValueError('redirect_limit')

    def fetch(self, url):
        if url in self.memory:
            return self.memory[url]
        record = {'url': url, 'final_url': url, 'accessed_at': datetime.now(timezone.utc).isoformat(),
                  'status': 'access_failed', 'blocks': [], 'metadata': {}, 'content_hash': '', 'snapshot_path': None}
        try:
            p = urlparse(url)
            robots_url = f'{p.scheme}://{p.netloc}/robots.txt'
            code, _, data, _ = self._get(robots_url)
            if code == 200:
                robots = RobotFileParser()
                robots.parse(data.decode('utf-8', errors='replace').splitlines())
                if not robots.can_fetch('RAGStakeholderResearch', url):
                    raise ValueError('robots_denied')
            elif code not in (404, 410):
                raise ValueError('robots_unavailable')
            code, headers, data, final_url = self._get(url)
            record['final_url'] = final_url
            # 다른 origin으로 이동하면 새 origin robots 정책을 다시 확인한 뒤 별도 요청.
            if urlparse(final_url).netloc != p.netloc:
                record = {**self.fetch(final_url), 'url': url}
            elif code in (401, 402, 403):
                record['status'] = 'paywall_or_forbidden'
            elif code != 200:
                record['status'] = f'http_{code}'
            elif 'html' not in headers.get('content-type', '').lower():
                record['status'] = 'unsupported_content_type'
            else:
                html = data.decode('utf-8', errors='replace')
                blocks, metadata = parse_html(html)
                record.update(blocks=blocks, metadata=metadata)
                if not blocks or sum(len(b['text']) for b in blocks) < 100:
                    record['status'] = 'empty_or_js_required'
                elif re.search(r'"isAccessibleForFree"\s*:\s*(?:false|"false")', html, re.I):
                    record['status'] = 'paywall'
                else:
                    record['status'] = 'ok'
                record['content_hash'] = digest(json.dumps(blocks, ensure_ascii=False, sort_keys=True))
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                path = self.cache_dir / (digest(url) + '.html')
                path.write_bytes(data)
                record['snapshot_path'] = str(path.resolve())
        except Exception as exc:
            record['status'] = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / (digest(url) + '.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        self.memory[url] = record
        return record
