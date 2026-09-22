"""고정 PDF 코퍼스 검증·파싱·청킹.

원문은 실행 중 내려받지 않는다. manifest와 로컬 파일의 SHA-256이 다르면 즉시 실패해
서로 다른 revision이 조용히 섞이는 것을 막는다.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


Technology = Literal["sw", "hw"]


class CorpusValidationError(ValueError):
    """코퍼스 파일 누락·변조·manifest 불일치."""


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    filename: str
    title: str
    revision: str
    published_at: str
    author_or_org: str
    url: str
    page_count: int
    sha256: str
    role: Literal["primary", "supporting"]
    technology: Technology
    approach: str

    def to_parent_meta(self, source_dir: Path) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "revision": self.revision,
            "source_type": "paper",
            "url": self.url,
            "local_path": str((source_dir / self.filename).resolve()),
            "page_count": self.page_count,
            "sha256": self.sha256,
            "license": None,
        }


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    doc_id: str
    title: str
    author_or_org: str
    url: str
    published_at: str
    role: Literal["primary", "supporting"]
    technology: Technology
    approach: str
    page: int
    locator: str
    text: str
    content_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_manifest_path() -> Path:
    return default_repo_root() / "data" / "technical" / "manifest.json"


def default_source_dir() -> Path:
    return default_repo_root() / "data" / "technical" / "sources"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\u00ad", "")).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path | None = None) -> list[SourceDocument]:
    manifest_path = Path(path or default_manifest_path())
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("manifest_version") != "technical-corpus-v1":
        raise CorpusValidationError("지원하지 않는 기술 코퍼스 manifest 버전입니다.")
    documents = [SourceDocument(**item) for item in payload.get("documents", [])]
    if not documents:
        raise CorpusValidationError("manifest에 문서가 없습니다.")
    if len({doc.doc_id for doc in documents}) != len(documents):
        raise CorpusValidationError("manifest의 doc_id가 중복됩니다.")
    return documents


def validate_corpus(
    documents: list[SourceDocument], source_dir: Path | None = None
) -> list[dict]:
    root = Path(source_dir or default_source_dir())
    checked: list[dict] = []
    for document in documents:
        path = root / document.filename
        if not path.is_file():
            raise CorpusValidationError(f"원문 PDF가 없습니다: {path}")
        actual = file_sha256(path)
        if actual != document.sha256:
            raise CorpusValidationError(
                f"SHA-256 불일치: {document.filename} "
                f"(expected={document.sha256}, actual={actual})"
            )
        checked.append({"doc_id": document.doc_id, "path": str(path), "sha256": actual})
    return checked


def _split_page(text: str, chunk_chars: int, overlap_chars: int) -> list[tuple[int, int, str]]:
    clean = normalize_text(text)
    if not clean:
        return []
    if chunk_chars <= overlap_chars or overlap_chars < 0:
        raise ValueError("chunk_chars는 overlap_chars보다 커야 합니다.")
    parts: list[tuple[int, int, str]] = []
    start = 0
    while start < len(clean):
        hard_end = min(len(clean), start + chunk_chars)
        end = hard_end
        if hard_end < len(clean):
            boundary = clean.rfind(". ", start + chunk_chars // 2, hard_end)
            if boundary > start:
                end = boundary + 1
        value = clean[start:end].strip()
        if value:
            parts.append((start, end, value))
        if end >= len(clean):
            break
        start = max(start + 1, end - overlap_chars)
    return parts


def parse_corpus(
    documents: list[SourceDocument],
    source_dir: Path | None = None,
    *,
    chunk_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[CorpusChunk]:
    """PDF를 페이지 단위로 읽고 locator를 보존한 청크를 만든다."""
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - 설치 안내 경로
        raise RuntimeError("PDF 파싱에는 pdfplumber가 필요합니다.") from exc

    root = Path(source_dir or default_source_dir())
    validate_corpus(documents, root)
    chunks: list[CorpusChunk] = []
    for document in documents:
        path = root / document.filename
        with pdfplumber.open(path) as pdf:
            if len(pdf.pages) != document.page_count:
                raise CorpusValidationError(
                    f"페이지 수 불일치: {document.filename} "
                    f"(expected={document.page_count}, actual={len(pdf.pages)})"
                )
            for page_number, page in enumerate(pdf.pages, start=1):
                for start, end, text in _split_page(
                    page.extract_text() or "", chunk_chars, overlap_chars
                ):
                    locator = f"p.{page_number}:chars:{start}-{end}"
                    raw_id = f"{document.doc_id}|{locator}|{text}"
                    chunk_id = "technical:chunk:" + hashlib.sha256(
                        raw_id.encode("utf-8")
                    ).hexdigest()[:16]
                    chunks.append(
                        CorpusChunk(
                            chunk_id=chunk_id,
                            doc_id=document.doc_id,
                            title=document.title,
                            author_or_org=document.author_or_org,
                            url=document.url,
                            published_at=document.published_at,
                            role=document.role,
                            technology=document.technology,
                            approach=document.approach,
                            page=page_number,
                            locator=locator,
                            text=text,
                            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        )
                    )
    return chunks

