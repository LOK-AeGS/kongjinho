# ISSUE

디렉터리 재구성(`refactor/agent-layout`) 이후 남은 문제 목록입니다.
팀 State 설계서가 나오면 1번을 먼저 해결하고, 2번은 설계서에서 함께 정합니다.

상태 표시: ⬜ 미해결 · ✅ 해결

---

## 1. State (최우선, 설계서 반영 대기)

### ⬜ 1-1. 부모 State가 두 개로 나뉘어 있음
- `graph/state.py`: `PipelineState` (최초 설계). 도메인 에이전트가 이 구조를 기준으로 합니다.
- `graph/team_state.py`: `EvaluationState` (팀 설계서 v0.3). 이해관계자 에이전트가 이 구조를 기준으로 합니다.
- 이해관계자는 `make_node(legacy=True)`로 PipelineState에도 붙일 수 있습니다.
- **할 일:** 설계서 기준으로 하나로 통일하고, 두 에이전트의 `node.py`를 거기에 맞춥니다.

### ⬜ 1-2. 근거(evidence) 형식과 reducer가 두 벌
| | 도메인 | 이해관계자 |
|---|---|---|
| ID 키 | `evidence_id` | `id` |
| 위치 키 | `locator` | `page_or_locator` |
| reducer | `agents/domain/rag/evidence.py`의 `merge_evidence` | `graph/team_state.py`의 `merge_evidence` |

- 두 에이전트 모두 `evidence_store`에 씁니다. 그런데 한 키에는 reducer를 하나만 붙일 수 있습니다.
- 이해관계자 reducer는 `incoming['id']`를 읽기 때문에, 도메인 근거가 들어오면 `KeyError`가 납니다.
- **할 일:** 근거 형식 하나와 reducer 하나를 `graph/`에 정의하고, 각 에이전트 `node.py`에서 그 형식으로 변환합니다.

### ⬜ 1-3. State에 없는 키는 조용히 버려짐
- LangGraph는 State에 선언되지 않은 키를 반환하면 **에러 없이 버립니다.** 직접 확인했습니다.
- 도메인 노드는 `domain_findings` 외에 `evidence_store`, `quality_by_perspective`, `search_log_by_perspective`도 반환합니다.
  `PipelineState`에는 이 세 키가 없어서, 지금 연결하면 도메인이 모은 근거가 사라집니다.
- **할 일:** 설계서의 State에 이 키들(또는 대체 키)을 reducer와 함께 선언합니다.

### ⬜ 1-4. `domain_findings` 모양이 `graph/state.py` 정의와 다름
- `graph/state.py`의 `DomainFindings`는 근거를 `evidence: list[Evidence]`로 findings 안에 담습니다.
- 도메인 에이전트는 `cited_evidence_ids`만 담고, 근거 본문은 `evidence_store`에 둡니다.
- 변환 함수 `to_team_findings()`는 `agents/domain/state.py`에 있습니다.
- **할 일:** 설계서에서 근거를 findings 안에 둘지, 공유 `evidence_store`에 둘지 정합니다.

---

## 2. 설계서에서 함께 정할 것

### ⬜ 2-1. 평가 도메인이 데이터센터로 고정됨
- 이해관계자: 도메인이 `datacenter`가 아니면 `ValueError`를 냅니다 (`agents/stakeholder/subgraph.py`의 `run_stakeholder`).
- 도메인: 프롬프트에 데이터센터가 고정되어 있습니다 (`agents/domain/prompts.py`).
- 반면 `PipelineState.request.domains`는 `list[str]`라서 여러 도메인을 받을 수 있게 되어 있습니다.
- **정할 것:** 도메인을 하나로 할지 여러 개로 할지.

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

### ⬜ 3-1. openai 3.x에서 실제 API 호출 미검증
- 도메인 lock은 `openai 3.x`, 이해관계자는 원래 `openai<3`로 고정되어 있었습니다.
- 통합 `requirements.txt`에서는 상한을 풀었습니다 (`openai>=2.0`).
- openai 3.16.2에서 오프라인 테스트 40개는 통과했습니다.
- 이해관계자의 실제 API 호출(`responses.create`, `responses.parse`)은 아직 3.x에서 확인하지 않았습니다.
- **할 일:** API 키를 넣고 `python -m scripts.run_stakeholder --max-queries 1 --rounds 1 --revisions 0`으로 작게 실행해 봅니다.

### ⬜ 3-1a. 이해관계자 실제 실행에서 검색이 실패함: 출력 토큰 한도 부족 (원인 확정)
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
- **할 일 (이해관계자 담당):**
  1. `backend.py`의 `search()`에서 `max_output_tokens=3000`을 8000 정도로 올립니다.
  2. `extract()`의 `max_output_tokens=10000`도 같은 이유로 부족할 수 있습니다. 원문 입력이 크면 추론이 길어지므로 확인합니다.
  3. 실패 시 에러 종류뿐 아니라 메시지와 `response.incomplete_details`도 검색 로그에 남깁니다.
     지금은 `ValueError`만 남아서 원인을 찾으려면 별도 진단 호출이 필요했습니다.
  4. 검색 결과 URL 끝에 `?utm_source=openai`가 붙어 옵니다. 근거 ID(URL 기반 해시)와 중복 판정에 영향이 없는지 확인합니다.

### ⬜ 3-2. lock 파일이 두 개
- `requirements.lock` (도메인), `requirements.lock.txt` (이해관계자)
- **할 일:** State 수정이 끝난 뒤 통합 환경에서 lock 파일 하나를 다시 만듭니다.

---

## 4. 검증

### ⬜ 4-1. 재구성 후 실제 API로 실행해 보지 않음
- 로직은 바꾸지 않았고, 오프라인 테스트 40개와 이해관계자 fixture 실행은 통과했습니다.
- 두 에이전트 모두 재구성 후 실제 API로는 실행하지 않았습니다.
- 이해관계자는 재구성 전에도 API 키 인증 오류 때문에 실제 검색 end-to-end가 확인되지 않은 상태였습니다.
- **할 일:** 키를 넣고 각 에이전트를 단독으로 한 번씩 실행합니다.

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

