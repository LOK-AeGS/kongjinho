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

### 1-2. 두 에이전트의 `node.py`가 아직 AppState 형식을 쓰지 않음 (최우선)

**✅ 도메인 (해결, 2026-09-22)** — `agents/domain/node.py`를 AppState 기준으로 다시 썼습니다.
`selected_tech["sw"]["name"]`/`request["as_of"]`를 읽고, `domain_findings`를
`PerspectiveFindings`(records: 요구사항 축 × 기술마다 `VerdictRecord` 하나, 최대 12건 /
claims / gaps)로, 근거를 `graph.state.Evidence`(`id`/`page_or_locator`/
`primary_or_secondary`/`direct_or_proxy`/`stance` 등)로 변환합니다.
`quality_by_perspective["domain"]`은 이제 guard/lint/judge 세 딕셔너리가 아니라
`QualityReport`(status/violations/warnings/checked_claim_ids) 하나입니다 — 결정적 코드
검증 3종을 없애고 분석 프롬프트의 자체 점검(self_check)으로 합쳤기 때문입니다
(`docs/DOMAIN_AGENT.md` "품질 검증" 절 참고). AppState에는 오류·수집 페이지 수를 담을
전용 필드가 없어 `run_meta["domain"]`에 남깁니다. `graph.build.build_graph(domain=...)`에
연결해 실제 API로 end-to-end 실행까지 확인했습니다(4-1 참고).

⬜ **이해관계자 (미해결)**

> 참고: 별도 간소화 버전 `agents/stakeholder_eval.py`(PR #6)는 AppState 형식으로 들어왔고, 부모 그래프(`main.py --live stakeholder`)는 이쪽을 씁니다.
> 시장(PR #4)·평가 종합(PR #5)·보고서(PR #7)도 AppState 형식입니다.


| | 이해관계자 (`agents/stakeholder/node.py`) |
|---|---|
| 입력 | v0.3: `selected_tech`, `domain`, `as_of_date`를 읽음 (`as_of_date`는 AppState에서 `request.as_of`)<br>legacy: `state["request"]["sw"]` → **KeyError** |
| 결과 키 | v0.3: `stakeholder_eval`, `errors` → AppState에 없어서 **버려짐**. `stakeholder_findings`로 반환해야 함 |
| 근거 | 거의 같음. `evidence_level`, `metric_tag` 추가, `claim_ids`/`perspectives`/`bindings`는 AppState에 없음 |
| 도메인 값 | AppState 기본값 `"datacenter_inference"`를 받으면 `datacenter`가 아니라서 **ValueError** |

- **할 일 (이해관계자 담당):** `node.py`에서 AppState를 읽고, `stakeholder_findings`를
  `PerspectiveFindings`로, 근거를 `graph.state.Evidence`로 변환해 반환합니다.
  `subgraph.py` 등 내부 로직은 바꾸지 않아도 됩니다. 도메인의 변환 예시(`_to_team_*`
  함수, `agents/domain/node.py`)를 참고할 수 있습니다.
- 변환 후 fixture로 돌려서 `evidence_store`와 `stakeholder_findings`가 실제로 채워지는지 확인합니다.

### ⬜ 1-3. `Evidence.content_hash`의 의미가 정해지지 않아 에이전트마다 다름 (팀 결정 필요)
- `content_hash`는 공유 키 `evidence_store` 안의 각 근거(`graph.state.Evidence`, `graph/state.py:114`)에 붙는 필드입니다.
- 설계서 §7.1 에는 `content_hash: str` 한 줄만 있고 **무엇을 해시하는지 정의가 없습니다.**
  바로 옆의 `id`는 "SHA-256(정규화 URL + 문서 내 위치 + 정규화 인용문)"으로 정의돼 있습니다.
- 설계서의 쓰임새:
  - §7.3 `n_evidence: int  # 서로 다른 content_hash 수`
  - §8.6.1 "그 근거가 같은 content_hash 1건이면 일치가 아니라 공유 근거(shared_evidence)"
  - §8.6.3 이중 계산 방지: "content_hash 기준 일치 판정"
  - §3.5 웹 검색 기록 항목에 포함
- 현재 구현:

  | 에이전트 | `content_hash` 값 | 위치 |
  |---|---|---|
  | 시장 | 인용문 하나의 해시 | `agents/market/state.py:162` |
  | 도메인 | 문서 전체(스냅샷)의 해시 (`content_sha256`) | `agents/domain/node.py:103`, `agents/domain/tools/evidence.py:74` |
  | 이해관계자 | 페이지 전체(블록 전체)의 해시 | `agents/stakeholder/subgraph.py:45` |

- **영향:** 평가 종합이 이 값으로 공유 근거, 매트릭스 근거 수(`n_evidence`), SX6 을 판단합니다.
  같은 논문을 인용해도 시장 쪽은 "다른 근거", 도메인 쪽은 "같은 근거"로 세어집니다.
  문서 단위일 때는 같은 논문의 장점 인용과 한계 인용이 SX6(같은 근거의 반대 stance)으로 잘못 잡힙니다.
- **제안:** "수집한 원문 문서(스냅샷) 전체의 SHA-256"으로 정의합니다.
  인용문 단위라면 `id`와 사실상 같은 값이 되어 필드를 따로 둘 이유가 약하고, §8.6.1 의 "같은 자료 = 근거 1건"과도 맞습니다. (추론이므로 팀 확인 필요)
- **할 일:** 팀이 정의를 확정해 `graph/state.py`의 `Evidence.content_hash`에 주석으로 적고, 다른 에이전트가 맞춥니다.
  평가 종합의 SX6 은 `content_hash`가 아니라 근거 `id` 기준으로 바꿉니다 (6-2).

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

### 2-3. 에이전트마다 LLM이 다름

**도메인: `gpt-4o`로 결정 (2026-09-22)** — 이전 `gpt-4o-mini`에서 변경. LangChain
`init_chat_model` 경유, 실제 API로 end-to-end 실행 확인함(4-1 참고).

⬜ 이해관계자: `gpt-5.5` (openai SDK 직접, Responses API) — 아직 gpt-4o로 통일할지
정하지 않았습니다. **정할 것:** 팀 공통 모델로 통일할지 여부. 비용에도 영향을 줍니다.

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

### ✅ 4-1. 재구성 후 실제 API로 실행해 보지 않음 (해결)
- 이해관계자: ✅ 2026-09-22 실제 API로 end-to-end 실행 확인 (3-1a 수정 후).
- 도메인: ✅ 2026-09-22 AppState 변환(1-2) 후 `python -m scripts.run_domain`과 노트북으로
  실제 API(gpt-4o + Tavily) 4회 실행 확인. 최종 노트북 재실행은 14개 코드 셀이
  모두 오류 없이 완료됐고, 근거 31건 / 주장 12건 / 판정 12건(6축×2기술),
  `status=partial`이었습니다. ACM·OpenReview·IEEE Xplore 등이 403(봇 차단)을
  반환했지만 그래프는 중단 없이 오류를 `run_meta["domain"].errors`에 기록하고
  진행했습니다.
- 3회 실행 중 품질 검증(self_check) 프롬프트를 두 번 다듬었습니다 — 근거가 경쟁 제품을
  다루는데 대상 기술 실측처럼 쓴 사례를 1차에서 발견 → 2차 수정에서 과잉 반응(정상 근거
  10건에 오탐) → 3차 수정에서 위반 3건으로 줄었고 전부 실제로 타당했습니다. 자세한 내용은
  `docs/DOMAIN_AGENT.md`의 "품질 검증" 절 참고.
- 같은 v3.2 프롬프트로 수행한 4차 최종 검증에서는 `needs_review` 위반 2건
  (`hw-accuracy-1`, `sw-power-1`의 근거 부족)을 보고했습니다. 실행별 결과 차이를
  감추지 않고 노트북 출력에 남겼습니다.

### ⬜ 4-2. 전체 그래프를 모든 실제 노드로 돌려보지 않음
- ✅ 2026-09-22 루트 `main.py`로 부모 그래프를 연결했습니다 (`docs/PARENT_GRAPH.md`).
  오프라인 실행에서 `technical → market + stakeholder + domain(병렬) → synthesis → report` 순서를 확인했습니다.
- ⬜ 기술 조사(①) 에이전트가 아직 없어 `graph/stubs.py`의 fixture 재생 노드로 채웁니다.
- **할 일:** 기술 조사 PR 이 들어오면 `main.py`에 연결하고, `--live` 전체 실행으로 end-to-end 를 확인합니다.

---

## 5. 작은 정리거리

### ✅ 5-1. 도메인 산출물 위치가 규칙과 다름 (해결)
- `agents/domain/tools/ablation.py`의 `--out`/`--artifacts` 기본값을 `outputs/domain/`
  아래로 옮겼습니다. `scripts/run_domain.py`도 `outputs/domain/<timestamp>/`에 저장합니다.

### ✅ 5-2. ablation 실행 안내가 실제 옵션과 다름 (해결)
- `agents/domain/tools/ablation.py` 맨 위 설명을 실제 옵션(`--cache`, 기본값
  `data/fetch_cache`)에 맞게 고쳤습니다.

---

## 6. 평가 종합 에이전트 (`agents/synthesis/`) 코드 리뷰 결과

2026-09-22 리뷰. 테스트는 모두 통과하지만, 합성 fixture 가 실제 에이전트 데이터와 달라서 드러나지 않은 문제입니다.
코드 위치와 설명은 `docs/SYNTHESIS_AGENT.md` 참고.

### ⬜ 6-1. SX7(TRL 입력 불일치)이 실제 데이터에서 거의 항상 걸림 🔴
- 위치: `agents/synthesis/relations.py:139-150`
- 지금 조건: 시장·이해관계자 record 의 pilot 이상 근거가 `technical_findings.input_evidence_ids`에 없으면 상충.
- 문제: 기술 조사(①)는 Pool A(D1·D3 논문)만 보므로 시장·이해관계자 웹 근거가 TRL 입력에 들어갈 수 없습니다.
- 설계서 §8.6.1 의 SX7 은 "TRL 판정 **이후** 시장·이해관계자 근거가 새로 추가됨"입니다.
- **할 일:** 근거의 `accessed_at`이 TRL 입력 근거의 수집 시각보다 늦은 경우로 조건을 바꿉니다.

### ⬜ 6-2. SX6(같은 근거의 반대 stance)가 문서 단위로 잘못 걸림 🔴
- 위치: `agents/synthesis/relations.py:176-186`, `agents/synthesis/matrix.py:29`(`content_key`)
- 문제: `content_hash`로 "같은 근거"를 판단하는데, 도메인·이해관계자는 이 값이 문서 단위입니다 (1-3).
  같은 논문의 장점 인용(p.2)과 한계 인용(p.11)이 SX6 으로 잡힙니다.
- 또 아무 관점도 인용하지 않은 근거까지 훑어서, 관련 record 가 빈 상충 항목이 생깁니다.
  시장 출력을 넣어 확인했을 때 실제로 발생했고, 그 문장은 C2 검사에서 제거되어 결과가 `needs_review`로 떨어졌습니다.
- **할 일:** SX6 은 근거 `id` 기준으로 판단하고, 관점이 실제 인용한 근거만 봅니다.

### ⬜ 6-3. SX4(수치 차이)가 연도를 값으로 읽고 단위를 무시함 🟡
- 위치: `agents/synthesis/relations.py:166`
- 인용문의 첫 숫자만 봐서 "In 2024 ... 15.8B by 2028"이면 2024 를 값으로 읽습니다. million / billion 도 구분하지 않습니다.
- **할 일:** 연도(4자리) 제외, 단위 정규화.

### ⬜ 6-4. SX5(시점 차이)가 `YYYY-MM` 날짜를 무시함 🟡
- 위치: `agents/synthesis/relations.py:71`(`_date`)
- `matrix.py`의 `_parse_date`는 `YYYY-MM`도 읽는데 relations 는 따로 만든 함수를 써서 버립니다. 설계서 §6.3 은 날짜를 `YYYY-MM`으로 기록합니다.
- **할 일:** `matrix._parse_date`를 같이 씁니다.

### ⬜ 6-5. LLM 서술의 재현성 없음 🟡
- 위치: `agents/synthesis/writer.py:114`, `writer.py:28`
- `temperature` 미지정이라 실행마다 문장이 달라지고, 실행 기록에 모델 이름만 남습니다 (설계서 §10.0 은 temperature·seed 기록 요구).
- LLM 에 넘기는 record 순서가 입력 순서 그대로라, 순서 교환 테스트가 서술 부분까지는 보장하지 않습니다.
- **할 일:** gpt-4.1 에 `temperature=0`, payload 정렬, `meta`에 temperature 기록.

### ⬜ 6-6. 작은 정확도·표시 문제 🟢
- `relations.py:230` 도메인에 "병행" 주장이 하나라도 있으면 대조표의 도메인 행이 전부 complement 로 표시됨
- `relations.py:208-212` ID 번호를 규칙별로만 세어 `sx1:hw:001` 다음이 `sx1:sw:002`, 정렬도 hw 가 먼저
- `review.py:24` C7 이 "TRL" 글자가 있을 때만 검사해서 "5-6단계"처럼 쓰면 매트릭스와 다른 값도 통과
- `node.py:20` 요청에 기준일이 없으면 `date.today()`를 써서 같은 입력도 날짜에 따라 결과가 달라짐
