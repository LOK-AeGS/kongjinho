# Multi-Agent Orchestration 과제 — 남은 작업 정리

> 착수 시점의 작업 목록이다. 아래 "현재 상태" 표는 그 시점(고정 DAG) 기준이다.

- 작성일: 2026-10-07
- 근거: 10/06·10/07 수업 전사, 과제 가이드(Agent Pattern / Mandatory Items / State Schema / 품질 평가 / Deliverables), 현재 코드(`kongjinho/`)
- **마감: 오늘(10/7) 퇴근 전** (늦어도 다음 주 수요일 평가 전. 강사는 "웬만하면 오늘"을 권장)

---

## 0. 현재 상태 요약

[graph/build.py](../graph/build.py)는 `technical → (market · stakeholder · domain 병렬) → synthesis → report`로 **순서가 고정된 DAG**입니다. 그래서 이번 과제가 요구하는 아래 항목이 아직 없습니다.

| 가이드 요구 | 현재 |
|---|---|
| `add_conditional_edges` 동적 라우팅 | 없음 (고정 `add_edge`) |
| 근거 충분성 판단 후 보고서 작성 | 없음 |
| 근거 부족 시 하위 에이전트 재작업 | 없음 (에이전트 내부 `--rounds`만 있음) |
| 보고서 후 품질 평가 노드 + 미달 시 루프 | 그래프 수준 노드 없음 (보고서 내부 validator만 있음) |
| 레이어드 State, `trace_id`, `step_count` 등 제어 메타 | 단일 `AppState`, 제어 메타 없음 |
| LangSmith 트레이싱 | 코드에 연동 없음 |
| 체크포인터 | `build_graph(checkpointer=)`만 받고 `main.py`에서 미사용 |
| 보고서 10장 이내 | 16개 섹션 → 초과 가능성 |

**이미 있어서 재사용할 수 있는 것:** 에이전트 6개, 관점 격리(`project_input`), 결정적 validator, `status=partial/failed` + `gaps`, 우열 어휘 차단(C1~C7), 근거 불균형 감시, 인용 원문 대조, `trace.json`.

---

## P0. 없으면 감점 (배점 약 75점)

### 1. 패턴 확정: Supervisor 추천 (패턴 정합성 20 + 동적 동작 20)

**추천 이유**
- 9/22 실행에서 technical은 `partial`(근거 0건), market은 `partial`(공백 8개)이었습니다. "근거 부족 → 해당 에이전트 재작업"을 보여줄 실제 데이터가 이미 있습니다.
- 시장·도메인 에이전트의 `--rounds`와 결과의 `gaps`가 재작업 요청의 연결 지점이 됩니다.
- Orchestrator-Workers는 관점 4개를 항상 그대로 띄우면 "고정 fan-out"으로 보일 위험이 있습니다.

**trade-off (README에 한 줄 기재)**: 병렬 실행이 사라지고 Supervisor 호출 비용이 늘어납니다. (강사: Supervisor 패턴은 비용이 가장 큰 약점)

**구현 체크리스트**
- [ ] `supervisor` 노드 추가. 모든 하위 에이전트는 실행 후 **supervisor로만** 복귀 (하위 에이전트 간 직접 통신 금지)
- [ ] supervisor가 State(수집된 관점, 관점별 status·gaps·근거 수)를 보고 `add_conditional_edges`로 다음 노드 결정 — **순서 하드코딩 금지**
- [ ] 근거 충분성을 평가한 뒤에만 synthesis/report로 이동 — **스텝 수 고정 방식 금지**
- [ ] 근거 부족 시 `rework_reason` + 대상 gaps를 담아 해당 에이전트에 재작업 요청
- [ ] 강사 주의: Supervisor는 Router와 다름. 결과에 책임을 지고 매 실행 후 점검한다.
- [ ] 강사 주의: 한쪽 관점으로 쏠리지 않도록 supervisor 판단 기준을 균형 있게 설계

> Orchestrator-Workers를 선택할 경우: 서브태스크 목록을 구조화해 State에 저장, `Send`로 동적 fan-out, 결과 누적 + synthesizer 집계, 일부 worker 실패 시 계속/재시도/제외 중 하나를 정하는 fallback.

### 2. State 레이어드 재설계 (State Schema 20)

강사가 이번 실습에서 가장 기대한 부분: "한 통으로 설계한 State를 레이어드로 한 번 들었다 놓기".

- [ ] **SupervisorState (제어용)**: `next`, `trace_id`, `step_count` / `max_steps`, 에이전트별 `retry_count`, `node_status`, `last_error`, `rework_reason`, 타임스탬프(강사가 프로덕션 관점에서 강조)
- [ ] **에이전트별 State (페이로드용)**: 관점별 결과. 에이전트는 서로 State를 직접 공유하지 않음
- [ ] 원본/요약 분리: 원본은 외부 저장소, State에는 **요약본 + ID**만 (강사 권장 4번 방식. reducer로 계속 누적하면 LLM이 압축하며 내용이 유실될 수 있음)
- [ ] 최종 보고서는 State에 직접 넣지 말고 **URI만** (현재 `report_sections`가 State 안에 있음)
- [ ] 병렬/누적 필드는 reducer 명시 (이미 `evidence_store` 등에 있음)
- [ ] 가이드 표 7개 항목마다 설계 근거를 **코드 주석 + README**에 한 줄씩

| 항목 | 정리할 내용 |
|---|---|
| 제어 vs 페이로드 분리 | 라우팅에 필요한 최소 상태는? |
| 관측성 위치 | 결정 로그(사유 포함)를 State에 둘지 외부로 뺄지 |
| 지속성 비용 | 체크포인트마다 State가 무한 증식하지 않는지 |
| 상관 | State와 외부 로그를 잇는 키(`trace_id`/`run_id`) |
| 재개/복구 | 중단 후 재개에 필요한 최소 상태(상태·에러·재시도) |
| 동시 처리 | 동시 쓰기 필드의 reducer |
| 종료 보장 | step/반복 상한 |

### 3. 품질 평가 노드 + 미달 시 루프 (15점)

권장: **3안(Hybrid)** = 코드 검사 + LLM Judge

- [ ] 보고서 생성 **다음 단계**에 `quality_eval` 노드 추가
- [ ] 필수 4항목
  - Groundedness: 주장 → 출처 추적 (인용 원문 대조, REFERENCE 연결 재사용)
  - 중립성: 기술 추천·우열 판정 없음 (C1~C7, 우열 어휘 차단 재사용)
  - 편향 통제: 단일 출처·유리한 근거 편중 없음 (기술별 근거 수 2배 차이 감시 재사용)
  - 관점 커버리지: 4개 관점(기술 성숙도·시장성·이해관계자·도메인 적용) 포괄
- [ ] LLM Judge는 Generator(`gpt-4o-mini`)와 **다른 모델**, 점수와 함께 사유 기록 (강사: "왜 85점인지" 근거 제시)
- [ ] 미달 시 report 재실행 또는 supervisor 경유로 해당 관점 재작업, **루프 상한 필수**
- [ ] 강사 지적: 이전 제출물 중 "특정 기술 우위 판정"이 남은 사례가 있었음 → 중립성 검사 특히 확인

### 4. 종료 보장과 재현성 (재현성 10)

- [ ] 전역 `max_steps`, 에이전트별 재작업 상한, 평가 루프 상한
- [ ] `python main.py --live all`로 끝까지 실행해 보고서가 실제로 생성되고 무한 루프 없이 종료되는지 확인
- [ ] 제출하는 trace와 코드 실행 결과가 일치해야 함

### 5. LangSmith 트레이싱 캡처 (필수 산출물)

- [ ] `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` 설정 (`.env.example`에 키 이름만 추가, 값은 넣지 않기)
- [ ] 실행 시 metadata/tags에 `trace_id` 전달
- [ ] **재작업이 최소 1번 일어난 실행**을 캡처 (동적 처리 확인이 목적)
- [ ] 긴 경로는 `tracing-1.png`, `tracing-2.png` … 순서로 분할

### 6. 평가 보고서 10장 이내 (Output 10)

- [ ] 현재 16개 섹션 → 10장 이내로 축소
- [ ] 필수 목차 **SUMMARY, REFERENCE** 유지
- [ ] 강사 의도: 분량을 늘리기보다 제한된 범위에 필요한 내용이 들어가는지 확인

---

## P1. 제출 형식

7. **브랜치**: 기존 repo에서 브랜치만 분리 (예: `feat/supervisor-pattern`). 새 repo 생성 금지.
8. **README 갱신** (가이드의 핑크색 항목은 필수)
   - Overview: 패턴, 선정 이유, 고정 순서와 다른 점(동적 처리)
   - Agents: Supervisor 추가
   - State Schema: 7개 항목
   - Architecture: `graph.get_graph().draw_mermaid_png()` 이미지로 교체
   - Tech Stack: Judge 모델 명시
   - Contributors: 개인별 역할 (PM·PL 제외)
9. **결정 로그는 외부로**: `{trace_id, node, decision, reason, ts}`를 `outputs/.../decisions.jsonl`에 적재 (기존 `trace.json` 확장)
10. **체크포인터**: `MemorySaver`/`SqliteSaver` 연결, `thread_id = trace_id`. [main.py](../main.py)에서 현재 미사용.
11. **테스트 수정**: [tests/graph/test_parent_graph.py](../tests/graph/test_parent_graph.py)가 고정 실행 순서를 검증하므로 그대로 두면 깨짐. "market이 partial이면 재작업으로 라우팅되는지"를 검증하는 오프라인 테스트로 교체.

## P2. 여유 있을 때

12. **보안**: `kongjinho/.env`가 커밋·zip에 들어가지 않는지 확인 (`.gitignore` 점검)
13. **작업용 코드 정리**: 강사가 "불필요한 작업용 코드가 많이 남아 있다"고 언급 (배점 외)
14. **ISSUE.md 잔여 항목**: `content_hash` 정의(1-3), TRL 어휘 불일치(1-4) 등
15. **(선택)** 선택하지 않은 패턴(Orchestrator-Workers)도 구현해 두 패턴의 차이와 적합한 상황 비교
16. **제출 파일**: `Agent_판교_6반_강유성+지승환+이효은+이산+이동영+안균승.zip` (Git 링크 + 트레이싱 PNG + 보고서 PDF) → 반별 Slack 스레드

---

## 권장 진행 순서

1. **팀 결정**: Supervisor로 갈지 확정 (오늘 가장 먼저)
2. 새 브랜치 생성
3. `build.py` + State 레이어 골격 (항목 1·2)
4. 품질 평가 노드 + 루프 (3), 종료 상한 (4)
5. 실제 실행으로 재작업이 발생하는지 확인
6. LangSmith 캡처 (5), 보고서 10장 축소 (6)
7. README 갱신 (8), 테스트 정리 (11)
8. zip 작성 후 제출

## 역할 분담 (초안, 팀에서 조정)

| 작업 | 담당 |
|---|---|
| Supervisor + `build.py` (1) | |
| State 레이어 설계 + 설계 근거 (2) | |
| 품질 평가 노드 + 루프 (3) | |
| LangSmith, 체크포인터, 결정 로그 (5, 9, 10) | |
| 보고서 10장 축소 (6) | |
| README, 테스트, 제출 zip (8, 11, 16) | |

## 수업에서 강조된 주의사항

- State는 DB/컨테이너가 아니라 **에이전트 간 읽기/쓰기 인터페이스 규격**으로 생각할 것. API 요청 스키마 짜듯이 정의하지 말 것.
- State의 `results` 류 키와 하위 에이전트의 `task_output`은 **다르다**(개별 결과 vs 집계). 혼동 금지.
- 병렬 처리 시: 같은 키 동시 쓰기 금지(reducer 또는 ID 분리), 일부 실패 정책을 설계 단계부터 정할 것, fan-in은 모든 분기가 끝날 때까지 대기하되 타임아웃을 둘 것.
- 마케팅성 표현("모든 걸 에이전트에게 위임") 지양.
- LLM 가드레일("하지 마")만 믿지 말고 중간 단위 테스트와 트레이싱으로 의도치 않은 동작을 점검할 것.
