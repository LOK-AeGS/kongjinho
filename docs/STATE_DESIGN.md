# AppState 설계 (Supervisor 패턴)

`graph/state.py` 의 `AppState` 가 팀 공통 State 다. 에이전트는 서로를 import 하지 않고 이 State 의 **자기 소유 키만** 반환한다.
(이전 PipelineState 설계 기록은 git 이력의 `STATE_DESIGN.md` 를 참고.)

## 1. 키 구성 — 페이로드와 제어 메타데이터

| 구분 | 키 | 쓰는 노드 | reducer |
|---|---|---|---|
| 입력 | `request`, `selected_tech`, `domain`, `corpus_manifest` | 초기화(`create_initial_state`) | - |
| 페이로드 | `technical_findings`, `market_findings`, `stakeholder_findings`, `domain_findings` | 각 관점 에이전트 | 관점별 독립 키 |
| 페이로드 | `evidence_store` | 네 관점 (병렬) | `merge_evidence_store` (id 병합, 기존 값 우선·빈 필드만 보충) |
| 페이로드 | `synthesis` | 평가 종합 | - |
| 페이로드 | `report_sections`, `references` | 보고서 | `merge_dict_right` / - |
| 기록 | `search_log_by_perspective`, `quality_by_perspective`, `run_meta`, `retries` | 관점·보고서 | `merge_dict_right` |
| 제어 | `trace_id`, `run_id`, `max_steps` | 초기화 | - |
| 제어 | `step_count`, `next_action`, `next_targets`, `last_error`, `final_status` | supervisor | 덮어쓰기 |
| 제어 | `rework_requests`, `node_status` | supervisor, 워커 래퍼 | `merge_dict_right` |
| 제어 | `quality_verdict`, `quality_iterations` | quality 노드, supervisor(무효화) | 덮어쓰기 |
| 제어 | `decision_log` | supervisor | `append_capped` (최근 20건) |

`ReworkRequest = {perspective, gaps, focus_queries, extra_rounds, requested_by}`. 워커 래퍼(`graph/workers.py`)가 입력 State 에
`rework_hint` 로 실어 주고 `request.max_search_rounds` 를 `extra_rounds` 만큼 늘린다. 기술 조사·시장·이해관계자·도메인·보고서가 이를 읽는다.

## 2. 가이드 7항목에 대한 설계

| 항목 | 설계와 근거 |
|---|---|
| 제어 vs 페이로드 분리 | Supervisor 는 제어 키와 각 관점의 `status`·근거 수·gap 만 읽는다(`graph/sufficiency.py`). 본문은 읽지 않으므로 결정이 설명 가능하고 체크포인트가 가볍다. |
| 관측성 위치 | 결정(행동·대상·사유·결정 주체 `llm/rule/fallback`)은 `logs/<trace_id>.jsonl` 과 LangSmith 로 낸다. State 의 `decision_log` 에는 최근 20건만 둔다. |
| 지속성 비용 | `decision_log` 상한, `search_log_by_perspective` 관점별 최근 30건, `evidence_store` 는 id 로 병합하고 재작업은 새로 찾은 근거만 반환, 보고서 전문은 `report_sections` 에 한 번만 둔다. |
| 상관 | `trace_id`·`run_id` 를 초기 State 에 한 번 부여하고 State · 결정 로그 · LangSmith `metadata` · 체크포인터 `thread_id` 에 같은 값을 쓴다. |
| 재개/복구 | 체크포인터(`thread_id=run_id`, sqlite 가 있으면 `data/checkpoints.sqlite`). 재개에 필요한 최소 상태: `node_status`(상태·오류·시도 횟수), `last_error`, `rework_requests`, `step_count`. 워커 예외는 크래시 대신 `node_status=failed`. |
| 동시 처리 | `Send` 병렬 디스패치. 동시에 쓰는 키에는 reducer 를 붙이고, 워커는 reducer 키와 자기 결과 키만 쓴다. |
| 종료 보장 | `max_steps`(14), 관점별 재작업 2회, 전체 재작업 4회, 품질 루프 2회, 단계 실패 재시도 4회, `recursion_limit` 50. 상한에 닿으면 새 재작업 없이 종합 → 보고서만 마치고 `final_status=degraded`. |

각 항목이 코드에서 성립하는지는 `tests/graph/test_supervisor.py` 의 `StateContractTest` 와 `TerminationTest` 가 확인한다.

## 3. 규칙

- 노드는 자기 소유 키만 부분 업데이트로 반환한다. 선언되지 않은 키는 LangGraph 가 버린다.
- LLM·검색 클라이언트 같은 런타임 객체는 State 가 아니라 `make_node()` 인자로 주입한다.
- 근거·주장 ID 에는 에이전트 이름을 접두사로 붙인다(`market:ev:…`, `stakeholder:claim:…`).
- 실패해도 예외를 밖으로 던지지 않고 `status` 를 `partial`·`failed` 로 표시하고 이유를 `gaps` 에 남긴다.
