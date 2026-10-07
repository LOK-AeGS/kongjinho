# Supervisor 패턴 전환 설계 (구현 명세)

> 구현 전에 쓴 설계 명세다(2026-10-07). 구현하면서 달라진 점(LLM 선택 + guard, Judge 모델, 품질 재작업 대상 등)은 [README](../README.md)와 코드가 기준이다.

- 브랜치: `feat/supervisor-pattern` (base: `feat/report-page-budget`)
- 목적: 과제 "Multi-Agent Orchestration"의 Supervisor 필수 항목, State Schema 7개 항목, 품질 평가 노드를 코드로 충족
- 원칙: **기존 에이전트 내부 코드(agents/technical, market, domain, stakeholder_eval, synthesis)는 수정하지 않는다.**
  조정 계층은 `graph/`에, 품질 평가는 새 에이전트 `agents/quality/`에 둔다. 보고서 에이전트는 피드백 입력만 추가한다.

---

## 1. 과제 필수 항목 ↔ 구현 매핑

| 필수 항목 | 구현 |
|---|---|
| 하위 에이전트는 Supervisor와만 통신, 하위 간 직접 통신 금지 | 모든 worker 노드의 나가는 엣지는 `supervisor` 하나. worker 간 엣지 없음 |
| State(수집된 관점, 근거 충분도)에 따라 `add_conditional_edges`로 분기, 순서 하드코딩 금지 | `graph.add_conditional_edges("supervisor", route, [...])`. `route`는 `state["next"]`만 읽고, `next`는 `supervisor` 노드가 State로부터 계산 |
| 근거 충분성 평가 후에만 보고서 작성, 스텝 수 고정 금지 | `assess_sufficiency()` 결과가 4관점 모두 `sufficient` 또는 `accepted_insufficient`일 때만 synthesis로 진행 |
| 근거 부족 시 해당 하위 에이전트에게 재작업 요청 | 부족 관점만 `rework` 지시와 함께 다시 디스패치(최대 `MAX_REWORK_PER_AGENT`회) |
| 보고서 후 품질 평가 노드, 미달 시 Loop | `report → supervisor → quality_eval → supervisor`; 불합격이면 피드백과 함께 `report` 재실행 또는 관점 재작업 |
| 종료 보장 | `step_count`/`max_steps`, 에이전트별 재작업 상한, 품질 루프 상한 |

## 2. 그래프 토폴로지

```
START → supervisor
supervisor ──(conditional: state["next"])──▶ technical | market | stakeholder | domain | synthesis | report | quality_eval | END
technical/market/stakeholder/domain/synthesis/report/quality_eval ──▶ supervisor
```

- `route(state) -> list[str] | str`: `state["next"]`가 비면 `END`. 여러 개면 **같은 superstep에서 병렬 실행**
  (market·stakeholder·domain 동시 디스패치). LangGraph에서 conditional edge가 리스트를 반환하면 병렬 실행되고,
  각 노드가 `supervisor`로 엣지를 가지므로 supervisor는 다음 superstep에 한 번 실행된다.
- `path_map`(가능한 목적지 목록)을 명시해 그래프 이미지에 모든 분기가 보이게 한다.
- 파일: `graph/build.py`에서 `build_graph(*, technical, market, stakeholder, domain, synthesis, report, quality_eval, checkpointer=None, policy=None)`.

## 3. State 설계 (레이어드)

**Layer 1 — `SupervisorState`** (`graph/state.py`): 부모 그래프 State. `AppState`를 상속해 기존 페이로드 계약을 그대로 두고
제어 메타를 추가한다.

**Layer 2 — 에이전트 내부 State**: 각 에이전트의 서브그래프 State(`agents/market/state.py` 등, 기존 그대로).
경계는 `make_node()`의 projection(`project_input`)과 `graph/workers.py`의 wrapper.

```python
class NodeStatus(TypedDict):
    status: Literal["pending", "running", "done", "failed"]
    attempts: int                 # 실행 횟수(재작업 포함)
    completed_step: int | None    # 완료한 supervisor step (staleness 판정용)
    last_error: str | None
    sufficiency: str | None       # sufficient | insufficient | accepted_insufficient (관점 노드만)
    updated_at: str               # ISO8601 UTC 타임스탬프

class ReworkDirective(TypedDict):
    reason: str                   # 사람이 읽는 재작업 사유
    focus: list[str]              # 부족한 criterion/gap 요약 (최대 5개)
    round: int                    # 몇 번째 재작업인지 (1부터)
    max_search_rounds: int        # 재작업 시 검색 라운드 상향값 (기본 2)
    feedback: list[str]           # report 재작업 시 품질 평가 지적 사항

class Decision(TypedDict):
    step: int
    next: list[str]
    reason: str
    ts: str

class EvalVerdict(TypedDict):     # agents/quality가 반환하는 구조화 판정
    passed: bool
    mode: Literal["hybrid", "code_only"]
    criteria: dict[str, dict]     # groundedness/neutrality/bias_control/coverage → {passed, score, code, llm, reasons}
    failed_criteria: list[str]
    feedback: list[str]           # report writer에 넘길 수정 지시 (최대 8개)
    rework_targets: list[str]     # 관점 재작업이 필요하면 관점 이름 (coverage 실패 시)
    judge_model: str | None
    evaluated_report_version: int

class SupervisorState(AppState, total=False):
    # ── 제어 메타 (라우팅·종료·재개에 필요한 최소치) ──
    trace_id: str                                            # ★ 외부 로그/LangSmith/checkpoint thread_id 상관 키
    step_count: int                                          # supervisor 실행 횟수 (supervisor만 씀)
    max_steps: int                                           # 종료 가드
    next: list[str]                                          # 다음 디스패치 대상 (supervisor만 씀)
    node_status: Annotated[dict[str, NodeStatus], merge_dict_right]   # 병렬 worker 동시 쓰기 → reducer
    rework: dict[str, ReworkDirective]                       # supervisor만 씀
    last_decision: Decision | None                           # 마지막 결정 1건만 (전체 로그는 외부 JSONL)
    report_version: int                                      # report 실행 횟수 = 품질 루프 카운터
    # ── 페이로드 추가분 ──
    eval_result: EvalVerdict | None
    artifacts: Annotated[dict[str, str], merge_dict_right]   # 대용량 산출물은 파일, State엔 URI만
```

`create_supervisor_state(...)`: `create_initial_state` 결과에 `trace_id=uuid4().hex[:12]` 기본값, `step_count=0`,
`max_steps=DEFAULT_MAX_STEPS(20)`, `next=[]`, `node_status={}`, `rework={}`, `last_decision=None`, `report_version=0`,
`eval_result=None`, `artifacts={}`를 채운다.

### 3.1 7개 항목 설계 근거 (코드 주석과 README에 같은 문장으로)

| 항목 | 설계 |
|---|---|
| 제어 vs 페이로드 분리 | 라우팅은 `node_status`·`rework`·`step_count`·`next`만 읽는다. 관점 결과(`*_findings`)는 supervisor가 충분도 계산에만 읽고 수정하지 않는다 |
| 관측성 위치 | 결정 전체 로그는 State 밖 `outputs/graph/<run>/decisions.jsonl`(`{trace_id, step, node, decision, reason, ts}`). State엔 `last_decision` 1건만. LangSmith에는 run metadata로 `trace_id` |
| 지속성 비용 | 최종 보고서 Markdown·PDF는 파일로 쓰고 State엔 `artifacts`의 URI만. 원문 본문은 기존 fetch_cache, 결정 로그는 JSONL. `last_decision`은 덮어쓰기로 체크포인트마다 커지지 않음 |
| 상관 | `trace_id` = LangGraph checkpoint `thread_id` = LangSmith metadata/tags = decisions.jsonl 키 |
| 재개/복구 | `node_status`(attempts·completed_step·last_error)와 `rework`만 있으면 supervisor가 다음 행동을 재계산 가능. `MemorySaver` 체크포인터를 기본 연결 |
| 동시 처리 | 병렬 worker가 함께 쓰는 키는 reducer: `evidence_store`(멱등 병합), `node_status`·`artifacts`·`quality_by_perspective`·`search_log_by_perspective`·`run_meta`(dict 병합). 그 외 키는 노드별 소유 |
| 종료 보장 | `step_count >= max_steps`면 강제 END(사유 기록), `MAX_REWORK_PER_AGENT=1`, `MAX_REPORT_VERSIONS=2`, 모든 분기가 유한 카운터를 소모 |

## 4. Supervisor 로직 (`graph/supervisor.py`)

순수 함수로 분리해 단위 테스트 가능하게 한다.

```python
@dataclass(frozen=True)
class SupervisorPolicy:
    max_rework_per_agent: int = 1
    max_report_versions: int = 2
    min_evidence: dict[str, int] = {"technical": 3, "market": 3, "stakeholder": 3, "domain": 3}
    min_coverage: float = 0.5          # basis가 direct|inferred인 record 비율
    rework_search_rounds: int = 2

def assess_sufficiency(perspective, findings, evidence_store, policy) -> tuple[str, str, list[str]]:
    """(verdict, reason, focus) — verdict ∈ {"sufficient","insufficient"}.
    - findings 없음/ status=="failed" → insufficient
    - status=="complete" → sufficient
    - 그 외(partial): n_evidence = records+claims가 인용한 evidence 중 store에 있는 고유 수,
      coverage = basis∈{direct,inferred} record 비율.
      n_evidence >= min_evidence[p] and coverage >= min_coverage → sufficient, 아니면 insufficient
    - reason 예: "partial, 근거 2건(<3), coverage 0.33(<0.5)"
    - focus: gaps의 "technology/criterion" 상위 5개
    """

def decide(state, policy) -> Decision-like dict + state updates:
```

결정 규칙(위에서부터 처음 맞는 규칙). **실행 순서를 리스트로 하드코딩하지 않고 State 조건으로만 결정한다.**

0. `step_count >= max_steps` → `next=[]`, reason="max_steps 도달로 종료"
1. technical이 미실행 → `["technical"]` (다른 관점의 공통 입력이 technical 요약이라는 **의존성** 때문, reason에 명시)
2. technical이 insufficient이고 재작업 여유 있음 → technical 재작업
3. market/stakeholder/domain 중 미실행이거나 (insufficient이고 재작업 여유 있음)인 관점 전부 → 병렬 디스패치.
   재작업 대상에는 `rework[p] = ReworkDirective(...)` 기록. 재작업 여유가 없으면 `sufficiency="accepted_insufficient"`로 확정(사유 기록)
4. synthesis가 미실행이거나 어떤 관점의 `completed_step`이 synthesis의 `completed_step`보다 크면(stale) → `["synthesis"]`
5. report가 미실행이거나 synthesis보다 오래됐으면 → `["report"]`
6. eval_result가 없거나 `evaluated_report_version < report_version` → `["quality_eval"]`
7. eval_result.passed → END(reason="품질 평가 통과")
8. 불합격:
   - `rework_targets`에 관점이 있고 그 관점 재작업 여유 있음 → 그 관점 재작업 (이후 4~6이 자연히 다시 돈다)
   - 아니고 `report_version < max_report_versions` → `rework["report"] = {feedback: eval_result.feedback, ...}`, `["report"]`
   - 아니면 END(reason="품질 루프 상한 도달, needs_review로 종료")

supervisor 노드 반환: `{"step_count": +1 된 값, "next": [...], "rework": {...}, "last_decision": {...},
"node_status": {대상: status="running"...}}` (accepted_insufficient 확정도 node_status로).
결정마다 `DecisionLogger.log(trace_id, step, decision, reason)` 호출(외부 JSONL, 주입 가능, 기본 no-op).

## 5. Worker wrapper (`graph/workers.py`)

```python
def as_worker(name: str, node_fn, *, artifact_dir: Path | None = None) -> Callable[[dict], dict]:
```

- 입력 view: `rework.get(name)`이 있으면 `request.max_search_rounds`를 `directive["max_search_rounds"]`로 올린 복사본 State를 넘긴다
  (agents/technical은 1~2만 허용하므로 min(2, ...)). report 재작업이면 view에 `report_feedback=directive["feedback"]`를 넣는다.
- 원본 노드 호출, 예외는 잡아서 `node_status[name]={status:"failed", last_error:repr(e)[:300], ...}`만 반환(그래프 중단 없음)
- 성공 시 노드 반환값 + `node_status[name]={status:"done", attempts:+1, completed_step:state["step_count"], updated_at:...}`
- report worker: 반환값의 `report_sections["final_markdown"]`/`["final_markdown_with_ids"]`를 `artifact_dir/report.md`,
  `artifact_dir/report_with_ids.md`로 쓰고 State에서는 그 두 키를 제거, `artifacts={"report_md": uri, "report_with_ids_md": uri, "report_pdf": run_meta의 pdf_path}`,
  `report_version`을 +1. artifact_dir가 None이면 파일 쓰기 없이 키를 그대로 둔다(테스트 호환).
- worker는 다른 worker의 결과 키를 쓰지 않는다(기존 키 소유 규칙 유지).

## 6. 품질 평가 에이전트 (`agents/quality/`)

`agents/quality/__init__.py`(make_node export), `checks.py`(코드 검사), `judge.py`(LLM judge), `node.py`, `prompts.py`.

입력: `artifacts["report_with_ids_md"]`(파일) 또는 `report_sections["final_markdown_with_ids"]`, `evidence_store`, `*_findings`.

### 6.1 코드 검사 (`checks.py`, 결정적)

| 기준 | 검사 | 통과 조건 |
|---|---|---|
| groundedness | 본문(SUMMARY~6장, REFERENCE 제외)의 사실 불릿/표 행 중 `〔근거: …〕` 인용이 있는 비율. "확인되지 않았다/미확인/분량 제한" 같은 메타 문장은 분모에서 제외. 인용 ID가 evidence_store에 모두 존재 | 비율 ≥ 0.8, 미존재 ID 0 |
| neutrality | 금지 표현: 승자, 추천, 권장, 압도, 월등, 우위, 열위, 더 낫, 우수하, 최고의, 최선의, 선택해야, 바람직하 | 0건 |
| bias_control | (a) 단일 출처 집중: 인용 수 기준 최대 출처 비율 ≤ 0.4 (출처 identity는 `agents.report.references.source_identity`) (b) SW/HW 인용 균형: 기술별 claim 인용 수 비율 ≤ 2.5 (c) 기술마다 counter stance 근거 ≥ 1 (없으면 실패가 아니라 경고 + 한계점에 언급됐는지 확인) | a·b 충족 |
| coverage | 4.1~4.4 섹션이 모두 있고 각 섹션 본문이 "확인되지 않았다"만으로 채워지지 않았으며 인용 ≥ 1 | 4개 모두 |

각 검사는 `{passed: bool, score: float(0~1), details: [...]}` 반환.

### 6.2 LLM judge (`judge.py`)

- `with_structured_output`로 4개 기준 각각 `{score: 1~5, passed: bool, reasons: str, problem_sentences: list[str]}`
- 입력: 보고서 Markdown(ID 인용판) + 관점별 status 요약. 근거 원문은 넣지 않는다(토큰).
- judge 모델은 generator(gpt-4o-mini)와 **다른 모델**. 기본값은 실험 후 확정(`QUALITY_JUDGE_MODEL` env, 임시 기본 "gpt-4.1-mini").
- 판정 사유를 반드시 남긴다("왜 그 점수인지").

### 6.3 Hybrid 결합

- 기준별 `passed = code.passed and (llm is None or llm.score >= 3)`
- 전체 `passed = all(기준.passed)`
- `mode = "code_only"`: API 키가 없거나 judge 미주입(오프라인)일 때
- `feedback`: 실패 기준의 code details + llm problem_sentences를 "섹션명: 수정 지시" 형태로 최대 8개
- `rework_targets`: coverage 실패로 빠진 관점(예: "market")

node 반환: `{"eval_result": verdict, "quality_by_perspective": {"quality_eval": {status, violations, warnings, checked_claim_ids: []}}}`

## 7. 보고서 에이전트 피드백 입력 (최소 변경)

- `normalize_state`가 `state.get("report_feedback") or []`를 `context["quality_feedback"]`로 담는다(NormalizedInput에 필드 추가).
- `section_payload`가 피드백이 있으면 `payload["quality_feedback"]`에 넣고, `build_section_prompt`가
  "이전 품질 평가 지적(반드시 반영): …" 줄을 추가한다. 결정적 writer는 무시해도 된다.

## 8. main.py 통합

- `build_nodes`가 `quality_eval` 노드도 만든다: `--live`에 `quality` 포함 또는 `all`이면 LLM judge, 아니면 code_only.
  `LIVE_CAPABLE`에 "quality" 추가.
- 모든 노드를 `as_worker(...)`로 감싼다(`artifact_dir=folder`).
- `initial_state`는 `create_supervisor_state` 사용, `--max-steps`(기본 20) 옵션.
- `run_graph`: `MemorySaver` 체크포인터 + `config={"configurable": {"thread_id": trace_id}, "run_name": "kv-cache-supervisor",
  "metadata": {"trace_id": trace_id}, "tags": [f"trace:{trace_id}"], "recursion_limit": 100}`.
- `DecisionLogger(folder / "decisions.jsonl")`를 supervisor에 주입.
- LangSmith: `.env`에 `LANGSMITH_API_KEY`가 있으면 `LANGSMITH_TRACING=true`, `LANGSMITH_PROJECT`(기본 "kv-cache-supervisor")를 환경에 설정. 키 값은 출력하지 않는다.
  `.env.example`에 `LANGSMITH_API_KEY=`, `LANGSMITH_PROJECT=kv-cache-supervisor`, `QUALITY_JUDGE_MODEL=` 추가.
- 결과 저장: `report.md`는 artifacts의 파일(없으면 기존 방식), `decisions.jsonl`, `summary.md`에 **라우팅 경로(supervisor 결정 목록과 사유)**, 재작업 횟수, 품질 평가 결과 표 추가.
- 콘솔: supervisor 결정마다 `  [step] supervisor → market + domain (사유)` 한 줄.
- `steps_view`/`AGENTS` 등은 supervisor·quality_eval이 섞여도 깨지지 않게 수정.

## 9. 테스트 (API 키 없이)

`tests/graph/test_parent_graph.py`를 supervisor 기준으로 갱신하고 `tests/graph/test_supervisor.py` 추가:

1. 오프라인 전체 실행이 오류 없이 END에 도달하고 `report_version >= 1`, `eval_result` 존재
2. 첫 결정은 technical, 그다음 결정은 market·stakeholder·domain 병렬(같은 step)
3. fixture에서 partial인 관점(stakeholder·domain)은 재작업 1회 디스패치되고, 그래도 부족하면 `accepted_insufficient`
4. 하위 에이전트 → 하위 에이전트 엣지가 없음(그래프 구조 검사: 모든 worker의 out-edge는 supervisor뿐)
5. `max_steps`를 작게 주면 강제 종료되고 last_decision.reason에 사유
6. 품질 평가 불합격(가짜 quality 노드 주입) → report 재실행, `max_report_versions` 후 종료
7. `assess_sufficiency` 단위 테스트(complete/partial 충분/partial 부족/failed)
8. worker 예외 → node_status failed, 그래프는 계속(재작업 후 accepted_insufficient)
9. 키 소유: 각 worker는 자기 키 + node_status/artifacts/report_version(report만)만 갱신
10. 품질 코드 검사 단위 테스트: 금지 표현 탐지, 인용 비율, 단일 출처 집중, 관점 누락
11. 결정 로그 JSONL에 trace_id·reason이 기록

기존 테스트(153개)는 계속 통과해야 한다. 실행: `python3 -m pytest tests -q --ignore=tests/agents/report/test_report_llm_integration.py`
