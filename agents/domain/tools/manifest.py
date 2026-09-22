"""실행 매니페스트: 이 결과가 어떤 조건에서 나왔는지 기록한다.

보고서에 "처리량 1.80배"를 인용하려면 그 근거를 언제 어떤 모델로 어떤 검색 결과에서
뽑았는지 되짚을 수 있어야 한다. 모델 버전이나 검색 캐시가 바뀌면 같은 질문에도
다른 답이 나오므로, 기록 없이는 보고서의 수치를 재현할 수 없다.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

TRACKED_PACKAGES = (
    "langgraph", "langchain", "langchain-core", "langchain-openai", "langchain-tavily",
    "langchain-community", "langchain-huggingface", "sentence-transformers", "torch",
    "faiss-cpu", "rank-bm25", "pdfplumber", "httpx", "pydantic",
)


@dataclass
class LLMRunInfo:
    provider: str
    model: str
    temperature: float
    max_tokens: int | None
    seed: int | None
    prompt_version: str


@dataclass
class ArtifactRecord:
    """산출물 위치와 체크섬. 같은 체크섬이면 같은 입력으로 만든 것이다."""

    name: str
    path: str
    sha256: str
    bytes: int


def file_checksum(path: Path) -> ArtifactRecord | None:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ArtifactRecord(
        name=path.name, path=str(path), sha256=digest, bytes=path.stat().st_size
    )


def _package_versions() -> dict[str, str]:
    import importlib.metadata as meta

    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = meta.version(name)
        except meta.PackageNotFoundError:
            versions[name] = "(미설치)"
    return versions


def _hardware() -> dict:
    info: dict = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:  # 메모리는 표준 라이브러리로 못 읽어 sysctl/proc 를 쓴다
        if platform.system() == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
            )
            info["ram_gb"] = round(int(out.stdout.strip()) / 1024**3, 1)
        elif platform.system() == "Linux":
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    info["ram_gb"] = round(int(line.split()[1]) / 1024**2, 1)
                    break
    except Exception:
        info["ram_gb"] = None

    try:
        import torch

        info["torch_backend"] = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            )
    except Exception:
        info["torch_backend"] = "unknown"
    return info


@dataclass
class RunManifest:
    run_id: str
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    finished_at: str | None = None
    llm: dict | None = None
    embedding: dict | None = None
    search: dict = field(default_factory=dict)
    policy: dict = field(default_factory=dict)
    artifacts: list[dict] = field(default_factory=list)
    packages: dict = field(default_factory=_package_versions)
    hardware: dict = field(default_factory=_hardware)
    git_commit: str | None = None

    def record_llm(self, info: LLMRunInfo) -> None:
        self.llm = asdict(info)

    def record_embedding(self, info: dict | None) -> None:
        self.embedding = info

    def record_artifact(self, record: ArtifactRecord | None) -> None:
        if record is not None:
            self.artifacts.append(asdict(record))

    def finish(self) -> "RunManifest":
        self.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return self

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path


def current_git_commit(repo_dir: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None
