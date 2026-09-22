"""평가 종합 서브그래프 내부 State. 부모 AppState 와 별도로 관리한다.

노드마다 자기 필드만 채운다.
  matrix    → status, input_hash, matrix, gaps, imbalance, retry_requests, limitations
  relations → cross_findings, contrast_table
  write     → summary_claims, (cross_findings 의 explanation)
  review    → summary_claims(검사 통과분), dropped_sentences, quality
"""

from __future__ import annotations

from typing import TypedDict

from graph.state import (
    Claim,
    CompletionStatus,
    ContrastRow,
    CrossFinding,
    Evidence,
    Gap,
    MatrixCell,
    PerspectiveFindings,
)


class SynthesisLocalState(TypedDict, total=False):
    # 입력 (node.py 가 부모 State 에서 투영)
    as_of: str
    tech_names: dict[str, str]  # {"sw": "MLA", "hw": "ITME"}
    findings: dict[str, PerspectiveFindings | None]
    evidence_store: dict[str, Evidence]

    # ① matrix
    status: CompletionStatus
    input_hash: str
    matrix: list[MatrixCell]
    gaps: list[Gap]
    imbalance: dict[str, object]
    retry_requests: list[dict[str, object]]
    limitations: list[str]

    # ② relations
    cross_findings: list[CrossFinding]
    contrast_table: list[ContrastRow]

    # ③ write / ④ review
    summary_claims: list[Claim]
    dropped_sentences: list[dict[str, object]]
    quality: dict[str, object]
    errors: list[str]
