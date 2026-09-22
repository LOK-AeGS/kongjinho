"""임시 노드: 아직 AppState 형식 노드가 없는 자리를 fixture 재생으로 채운다.

부모 그래프 연결을 먼저 확인하기 위한 것이다. 팀원 PR 이 들어오면 main.py 에서
해당 자리를 실제 make_node() 로 한 줄씩 바꾸고, 다 바뀌면 이 파일을 지운다.

  replay_node("technical")  → technical_findings + 그 관점이 인용한 근거만 evidence_store 로
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

# 평가 종합 테스트용 합성 AppState. D1·D3 인용 외의 URL·수치는 가짜 자료다 (_note 참고).
DEFAULT_FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "agents" / "synthesis" / "fixtures" / "appstate_sample.json"


def load_fixture(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or DEFAULT_FIXTURE).read_text(encoding="utf-8"))


def cited_evidence(findings: dict | None) -> set[str]:
    """관점 결과가 실제로 인용한 근거 ID (records + claims)."""
    findings = findings or {}
    ids = {e for r in findings.get("records") or [] for e in r.get("evidence_ids", [])}
    ids |= {e for c in findings.get("claims") or [] for e in c.get("evidence_ids", [])}
    return ids


def replay_node(perspective: str, fixture: dict):
    """fixture 의 {perspective}_findings 와 그 관점이 인용한 근거를 그대로 돌려주는 노드."""
    key = f"{perspective}_findings"
    findings = fixture.get(key)
    store = fixture.get("evidence_store") or {}
    evidence = {eid: store[eid] for eid in sorted(cited_evidence(findings)) if eid in store}

    def node(state: dict) -> dict:
        return {key: copy.deepcopy(findings), "evidence_store": copy.deepcopy(evidence)}

    node.__name__ = f"replay_{perspective}"
    return node
