# ISSUE

디렉터리 재구성(`refactor/agent-layout`) 이후 남은 문제 목록입니다.
공통 State(`graph/state.py` AppState)는 확정됐습니다. 1-2(각 에이전트 변환)를 먼저 해결합니다.

상태 표시: ⬜ 미해결 · ✅ 해결

---

## 1. State

### ✅ 1-1. 부모 State가 두 개로 나뉘어 있음 (해결)
- 팀 공통 State를 `graph/state.py`의 `AppState`로 통일했습니다. `graph/team_state.py`(v0.3 EvaluationState)는 삭제했습니다.
- 근거 형식(`Evidence`)과 `evidence_store` reducer(`merge_evidence_store`)도 `graph/state.py` 하나로 정했습니다.
- 이해관계자 내부에서 근거 풀을 합칠 때 쓰던 reducer는 `agents/stakeholder/evidence.py`로 옮겼습니다.

### ⬜ 1-2. 두 에이전트의 `node.py`가 아직 AppState 형식을 쓰지 않음 (최우선)
State 파일만 교체했고, 각 에이전트의 입출력 변환은 아직 예전 형식입니다.
**지금 AppState 그래프에 두 노드를 그대로 연결하면 동작하지 않습니다.**
LangGraph는 State에 없는 키를 반환하면 에러 없이 버리므로, 결과가 조용히 사라질 수 있습니다.

| | 도메인 (`agents/domain/node.py`) | 이해관계자 (`agents/stakeholder/node.py`) |
|---|---|---|
| 입력 | `state["request"]["sw"]["name"]`을 읽음 → AppState에서는 `selected_tech["sw"]`라 **KeyError** | v0.3: `selected_tech`, `domain`, `as_of_date`를 읽음 (`as_of_date`는 AppState에서 `request.as_of`)<br>legacy: `state["request"]["sw"]` → **KeyError** |
| 결과 키 | `domain_findings` (claims/fits/cited_evidence_ids) → `PerspectiveFindings`(records/claims/gaps)로 변환 필요 | v0.3: `stakeholder_eval`, `errors` → AppState에 없어서 **버려짐**. `stakeholder_findings`로 반환해야 함 |
| 근거 | `evidence_id`/`locator` 형식 → `id`/`page_or_locator` 등 `Evidence` 형식으로 변환 필요 | 거의 같음. `evidence_level`, `metric_tag` 추가, `claim_ids`/`perspectives`/`bindings`는 AppState에 없음 |
| 품질 | `quality_by_perspective`에 guard/lint/judge 딕셔너리 → `QualityReport`(status/violations/warnings/checked_claim_ids)로 변환 필요 | 쓰지 않음 |
| 도메인 값 | 프롬프트에 데이터센터 고정 | AppState 기본값 `"datacenter_inference"`를 받으면 `datacenter`가 아니라서 **ValueError** |

- **할 일 (각 에이전트 담당):** `node.py`에서 AppState를 읽고, `*_findings`를 `PerspectiveFindings`로, 근거를 `graph.state.Evidence`로 변환해 반환합니다.
  `subgraph.py` 등 내부 로직은 바꾸지 않아도 됩니다.
- 변환 후 두 노드를 AppState 그래프에 연결해 fixture로 돌려서, `evidence_store`와 `*_findings`가 실제로 채워지는지 확인합니다.

---

## 2. 설계서에서 함께 정할 것

### ⬜ 2-1. 평가 도메인이 데이터센터로 고정됨
- 이해관계자: 도메인이 `datacenter`가 아니면 `ValueError`를 냅니다 (`agents/stakeholder/subgraph.py`의 `run_stakeholder`).
- 도메인: 프롬프트에 데이터센터가 고정되어 있습니다 (`agents/domain/prompts.py`).
- AppState는 `domain: str` 하나이고 기본값이 `"datacenter_inference"`입니다. 도메인은 하나로 정해졌습니다.
- **할 일:** 이해관계자의 `datacenter` 검사를 AppState 값에 맞춥니다 (1-2와 함께).

### ⬜ 2-2. 이해관계자 노드가 State의 반복 한도를 읽지 않음
- v0.3 노드(`stakeholder_agent`)는 검색 라운드, 수정 횟수, 질의 수를 `default_request()`의 기본값으로 씁니다.
- 도메인은 `request.max_search_rounds`를 State에서 읽습니다.
- **정할 것:** 반복 한도를 State에서 받을지 여부. 받는다면 키 이름도 정합니다.

### ⬜ 2-3. 에이전트마다 LLM이 다름
- 도메인: `gpt-4o-mini` (LangChain `init_chat_model` 경유)
- 이해관계자: `gpt-5.5` (openai SDK 직접, Responses API)
- **정할 것:** 팀 공통 모델로 통일할지 여부. 비용에도 영향을 줍니다.

### ⬜ 2-4. 수집량 한도의 기준이 다름
- 도메인: 웹 문서를 누적 200페이지까지로 제한합니다 (`PAGE_BUDGET`).
- 이해관계자: 페이지 한도 없이 검색 질의 수(기본 8회)로만 제한합니다.
- **정할 것:** 과제의 200페이지 제한을 웹 수집에도 적용할지, 에이전트별로 적용할지 전체 합산으로 적용할지.

---

## 3. 의존성

### ✅ 3-1. openai 3.x에서 실제 API 호출 (이해관계자 확인 완료)
- 도메인 lock은 `openai 3.x`, 이해관계자는 원래 `openai<3`로 고정되어 있었습니다.
- 통합 `requirements.txt`에서는 상한을 풀었습니다 (`openai>=2.0`).
- openai 3.16.2에서 오프라인 테스트 40개는 통과했습니다.
- 2026-09-22 openai 3.16.2에서 이해관계자의 실제 API 호출(`responses.create`, `responses.parse`)이 정상 동작함을 확인했습니다.
  (처음 실패는 SDK 버전이 아니라 3-1a의 토큰 한도 문제였습니다.)

### ✅ 3-1a. 이해관계자 실제 실행에서 검색이 실패함: 출력 토큰 한도 부족 (해결)
- 2026-09-22 실제 실행 2회 모두 실패했습니다: `python -m scripts.run_stakeholder --ask-key --max-queries 1 --rounds 1 --revisions 0`
  결과는 `상태: failed`, `오류: sw:competitor:r1: ValueError`였습니다.
- API 키와 연결은 정상입니다. 에러는 `agents/stakeholder/backend.py`의 `unpack_search()`가 내는 `search_incomplete`입니다.
- **원인:** 검색 요청의 `max_output_tokens=3000`이 부족합니다. `gpt-5.5`는 추론 토큰도 이 한도 안에서 씁니다.
  실패한 질의를 그대로 넣고 한도만 바꿔 비교한 결과입니다.

  | max_output_tokens | status | 출력 토큰 (추론) | 결과 |
  |---|---|---|---|
  | 3000 (현재) | `incomplete` (`reason='max_output_tokens'`) | 3000 (1336) | 검색 2회는 완료됐지만 답변 도중 잘려서 `search_incomplete` |
  | 8000 | `completed` | 1681 (1030) | 성공, URL 75개 |

  짧은 영어 질의는 3000 안에서 성공했지만(2869 사용), 실제 질의는 한글 그룹명 등이 붙어 더 길어서 한도를 넘습니다.
- **해결:** `backend.py`의 `search()`에서 `max_output_tokens`를 3000 → 8000으로 올렸습니다.
  수정 후 실제 실행(`--max-queries 1 --rounds 1 --revisions 0`)에서 검색 1회, 원문 5개 수집, 입장 6개 확인까지 완료했습니다.
  (`상태: partial`은 질의 1회로 8개 기술·그룹 조합 중 7개를 조사하지 않았기 때문이며 정상입니다.)
- **남은 할 일 (이해관계자 담당):**
  1. `extract()`의 `max_output_tokens=10000`도 같은 이유로 부족할 수 있습니다. 원문 입력이 크면 추론이 길어지므로 확인합니다.
  2. 실패 시 에러 종류뿐 아니라 메시지와 `response.incomplete_details`도 검색 로그에 남깁니다.
     지금은 `ValueError`만 남아서 원인을 찾으려면 별도 진단 호출이 필요했습니다.
  3. 검색 결과 URL 끝에 `?utm_source=openai`가 붙어 옵니다. 근거 ID(URL 기반 해시)와 중복 판정에 영향이 없는지 확인합니다.

### ⬜ 3-2. lock 파일이 두 개
- `requirements.lock` (도메인), `requirements.lock.txt` (이해관계자)
- **할 일:** State 수정이 끝난 뒤 통합 환경에서 lock 파일 하나를 다시 만듭니다.

---

## 4. 검증

### ⬜ 4-1. 재구성 후 실제 API로 실행해 보지 않음
- 로직은 바꾸지 않았고, 오프라인 테스트 40개와 이해관계자 fixture 실행은 통과했습니다.
- 이해관계자: ✅ 2026-09-22 실제 API로 end-to-end 실행 확인 (3-1a 수정 후).
- 도메인: ⬜ 재구성 후 실제 API로 실행하지 않았습니다.
- **할 일:** 키를 넣고 도메인 노트북을 한 번 실행합니다.

### ⬜ 4-2. 전체 그래프를 한 번도 돌려보지 않음
- `graph/build.py`는 있지만 실행 진입점(`main.py`)이 없습니다.
- technical, market, synthesis, report 에이전트가 아직 없습니다.
- **할 일:** State 확정 후 `main.py`를 만들고, 없는 에이전트는 임시 노드로 채워서 오프라인으로 전체 흐름을 돌려봅니다.

---

## 5. 작은 정리거리

### ⬜ 5-1. 도메인 산출물 위치가 규칙과 다름
- `outputs/ablation.json`, `outputs/run_manifest.json`이 `outputs/` 바로 아래에 있습니다.
- 규칙 16번은 `outputs/<에이전트>/`입니다.
- 코드에서 기본 경로를 쓰는 곳(`agents/domain/evaluation/ablation.py`, 노트북)을 함께 고쳐야 합니다.

### ⬜ 5-2. ablation 실행 안내가 실제 옵션과 다름
- `agents/domain/evaluation/ablation.py` 맨 위 설명에는 `--corpus data/corpus`라고 되어 있습니다.
- 실제 옵션은 `--cache`(기본값 `data/fetch_cache`)이고, `--corpus` 옵션은 없습니다.
- **할 일:** 도메인 담당자가 설명 문구를 실제 옵션에 맞게 고칩니다.

