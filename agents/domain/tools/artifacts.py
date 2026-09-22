"""색인 산출물 저장: 파싱 청크, FAISS, BM25, id 매핑.

인덱스를 메모리에만 두면 같은 수치를 다시 못 만든다. 청킹 파라미터나 파서 설정이 바뀌면
검색 결과가 달라지는데, 그때 쓴 청크가 남아 있지 않으면 어느 쪽이 원인인지 가릴 수 없다.
체크섬을 함께 남기는 이유는 "그때 그 인덱스가 맞다"를 확인하기 위해서다.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

from agents.domain.tools.index import BM25Index, Chunk, DenseIndex


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(root: Path) -> str:
    """디렉터리 전체의 내용 해시. 파일명과 내용을 정렬해 순서에 무관하게 만든다."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(_sha256_file(path).encode("utf-8"))
    return digest.hexdigest()


@dataclass
class SavedArtifact:
    name: str
    path: str
    sha256: str
    bytes: int
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name, "path": self.path, "sha256": self.sha256,
            "bytes": self.bytes, "note": self.note,
        }


def save_chunks(chunks: list[Chunk], out_dir: Path) -> SavedArtifact:
    """파싱 청크와 id 매핑을 한 파일에 담는다. chunk_index 가 색인 내 위치와 같다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "parsed_chunks.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(
                    {
                        "chunk_index": chunk.chunk_index,
                        "url": chunk.url,
                        "title": chunk.title,
                        "locator": chunk.locator,
                        "chars": len(chunk.text),
                        "text": chunk.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return SavedArtifact(
        name="parsed_chunks", path=str(path), sha256=_sha256_file(path),
        bytes=path.stat().st_size, note=f"{len(chunks)} chunks, id mapping 포함",
    )


def save_bm25(index: BM25Index, out_dir: Path) -> SavedArtifact:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bm25.pkl"
    # rank_bm25 객체는 학습된 통계를 들고 있어 그대로 직렬화하면 재구축 없이 되살릴 수 있다.
    path.write_bytes(pickle.dumps(index._bm25))
    return SavedArtifact(
        name="bm25_index", path=str(path), sha256=_sha256_file(path),
        bytes=path.stat().st_size, note="rank_bm25 BM25Okapi pickle",
    )


def save_faiss(index: DenseIndex, out_dir: Path) -> SavedArtifact | None:
    if index._store is None:
        return None
    out_dir = Path(out_dir) / "faiss"
    out_dir.mkdir(parents=True, exist_ok=True)
    index._store.save_local(str(out_dir))
    total = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    return SavedArtifact(
        name="faiss_index", path=str(out_dir), sha256=_sha256_tree(out_dir),
        bytes=total, note=f"model={index.run_info.model_name}, dim={index.run_info.dimension}",
    )


def describe_cache(cache_dir: Path, name: str) -> SavedArtifact:
    """검색·본문 캐시는 개별 파일이 많아 디렉터리 단위 해시로 기록한다."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return SavedArtifact(name=name, path=str(cache_dir), sha256="", bytes=0, note="없음")
    files = [p for p in cache_dir.rglob("*") if p.is_file()]
    return SavedArtifact(
        name=name, path=str(cache_dir), sha256=_sha256_tree(cache_dir),
        bytes=sum(p.stat().st_size for p in files), note=f"{len(files)} files",
    )
