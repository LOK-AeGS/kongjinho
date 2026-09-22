# 평가 종합 에이전트 (⑤ synthesis)

설계서 v0.8 §8.6 "평가 종합 에이전트"의 구현입니다.
네 관점(기술·TRL, 시장, 이해관계자, 도메인)의 결과를 처음으로 한자리에 모아 대조합니다.
**새 검색·새 사실을 만들지 않고, 관점별 판정을 고쳐 쓰지 않으며, 종합 점수·순위를 만들지 않습니다** (§2.1).

---

## 1. 입력과 출력

| 구분 | AppState 키 | 내용 |
|---|---|---|
| 읽음 | `request` | 기준일(`as_of`) |
| 읽음 | `selected_tech` | 기술 이름(`short_name`), 문장에 [MLA]·[ITME]로 표기 |
| 읽음 | `technical_findings`, `market_findings`, `stakeholder_findings`, `domain_findings` | 관점별 `PerspectiveFindings` (records / claims / gaps) |
| 읽음 | `evidence_store` | 인용 quote, 날짜, 증거 수준, content_hash |
| **씀** | `synthesis` | `SynthesisResult` 하나 |

다른 관점의 결과를 고치지 않고, `synthesis` 외의 키는 쓰지 않습니다.

---

## 2. 변경 내용: 그래프를 10개 노드에서 4개로 줄임

### 원래 설계 그림 (§8.6)

```
입력 → 결과 완결성 검사 → 비교 매트릭스 생성 → 관점 간 관계 분석
     → [보완 | 상충 | 일치] → 근거 강도·불확실성 반영 → 중립성 검토
     → 검토 통과? ─ 수정 필요 → 결론 표현·범위 수정 → (중립성 검토로 돌아감)
                  └ 통과 또는 한도 → 출력
```

### 구현한 그래프

```
START → ① matrix → ② relations → ③ write → ④ review → END
          (코드)       (코드)        (LLM)     (코드 + 위반 문장만 LLM 1회)
```

| | 원래 그림 | 구현 |
|---|---|---|
| 노드 | 10개 | **4개** |
| 분기 | 2곳 (세 갈래, 통과 여부) | **0** |
| 루프 | 1개 | **0** |

### 줄인 이유

1. **보완 / 상충 / 일치 세 갈래 → ② relations 하나.**
   경로를 고르는 분기가 아니라, 모든 레코드 쌍을 분류하는 작업입니다. 결과도 `CrossFinding.kind` 하나에 담기므로 노드를 셋으로 나눌 이유가 없습니다.
2. **완결성 검사 + 매트릭스 생성 + 근거 강도·불확실성 반영 → ① matrix 하나.**
   셋 다 분기 없는 코드 계산입니다. 근거 강도(`basis`), 증거 수준, 근거 수, stance 는 매트릭스 셀(`MatrixCell`) 자체에 들어가므로 "근거 강도 반영"은 별도 단계가 필요 없습니다. 완결성 검사는 결과를 거르지 않고 `gaps`·`limitations`·`retry_requests`에 기록만 합니다(`partial`도 정상 산출물, §2.2.7).
3. **중립성 검토 루프 → ④ review 안에서 처리.**
   설계 규칙이 "위반한 문장만 1회 재생성 → 그래도 위반이면 제거하고 `dropped_sentences`에 기록"(§8.6.2)입니다. 문장 단위·최대 1회라 그래프 루프와 "결론 수정" 노드가 필요 없습니다.

설계서의 "매트릭스 생성과 관계 분석은 코드가, 서술은 LLM이"(§8.6)와 같은 분담이고, §10의 6번("⑤의 매트릭스·상충 규칙을 LLM 없이 먼저 완성하고 fixture로 테스트")을 그대로 따를 수 있습니다.

---

## 3. 파일 구조

```
agents/synthesis/
├── __init__.py     make_node 만 노출
├── node.py         AppState → 내부 State 투영, 결과 → {"synthesis": SynthesisResult}
├── subgraph.py     4단계 직선 그래프, ③ write / ④ review 노드
├── state.py        SynthesisLocalState (내부 전용)
├── matrix.py       ① 완결성 검사 + 비교 매트릭스 + 불균형 집계
├── relations.py    ② SX1~SX7, 공유 근거·일치·보완, SW/HW 대조표
├── writer.py       ③ 서술기 2종 (TemplateWriter, OpenAIWriter), LLM 입력 구성
├── prompts.py      ③ 서술·재생성 프롬프트
├── schemas.py      OpenAIWriter 구조화 출력 스키마
├── review.py       ④ C1~C7 검사, 1회 재생성, 제거
└── rules.py        규칙 상수 (조건 필수 수치, 금지 수치, 우열 어휘, 임계값)

scripts/run_synthesis.py                          단독 실행
tests/agents/synthesis/test_synthesis.py          테스트 36개 (LLM·네트워크 없음)
tests/agents/synthesis/fixtures/appstate_sample.json   합성 AppState 입력
```

`.gitignore`에 실행 산출물 폴더 `outputs/synthesis/` 한 줄을 추가했습니다.

---

## 4. 단계별 동작

### ① matrix: 완결성 검사 + 비교 매트릭스

| 검사 | 처리 |
|---|---|
| 관점 결과가 `None` | `gaps`에 "관점 결과 없음", `retry_requests`에 추가 |
| `status=failed` | `retry_requests`와 `limitations`에 기록 |
| `status=partial` | `limitations`에 기록 |
| record 필수 필드 누락 | 해당 record 제외, `limitations`에 기록 |
| `evidence_store`에 없는 근거 ID | 그 ID만 제외하고 기록 |
| `basis=direct`인데 유효한 근거 없음 | 기록 |
| 기준일 이후 발행된 근거 | 기록 |
| 기술·시장·이해관계자 근거에 `accessed_at` 없음 | 기록 (§6.4 시간 규칙) |

- 매트릭스 셀은 record 하나당 하나입니다. `n_evidence`는 **서로 다른 content_hash 수**입니다(같은 자료를 두 번 세지 않음).
- 불균형(`imbalance`): 기술별 근거 수(중복 제거)와 basis 분포. `not_applicable`은 제외하고, 2배 이상 차이 나면 `limitations`에 기록합니다(§6.4).
- `status`: 관점 결과가 하나도 없으면 `failed`, 빠진 관점·공백·partial 이 있으면 `partial`, 아니면 `complete`.
- 정렬: 입력 순서와 무관하게 `sw → hw`, `technical → market → stakeholder → domain` 순서로 고정합니다(제시 순서 편향 방지).

### ② relations: 관점 간 관계

같은 기술 안에서 **서로 다른 관점**의 record 쌍을 비교합니다. record 의 방향은 `stance_counts`의 support 와 counter 중 많은 쪽입니다(같으면 neutral).

| kind | 판정 조건 |
|---|---|
| `shared_evidence` | 두 record 가 같은 content_hash 근거를 씀. **일치로 세지 않음** (이중 계산 방지, §8.6.1) |
| `agreement` | 공유 근거가 없고, 같은 `assessment_vocab` + 같은 방향(neutral 제외) |
| `complement` | 공유 근거가 없고, 다른 `assessment_vocab` + 같은 방향 (§2.2.6: "시장 adopted + 도메인 suitable"은 일치가 아니라 보완). 또는 `technology="both"` 주장이 SW·HW 병행을 직접 언급 |

상충 규칙(`kind=conflict`, `detected_by=rule`)은 설계서 §8.6.1을 이렇게 코드로 옮겼습니다.

| 규칙 | 구현한 탐지 조건 |
|---|---|
| SX1 증거 수준 | 시장 record 의 증거 수준이 `announcement` 이상인데, TRL record 값의 상한이 7 미만 (운영 환경 단계 근거 없음) |
| SX2 범위 | 기술군(`class`) record 와, 다른 관점의 대상 기술(`direct`) record 가 다른 방향. **basis 가 unknown·not_applicable 인 record 는 제외**, class record 하나당 한 건으로 묶음 |
| SX3 조건 탈락 | 관점 주장에 조건 필수 수치(93.3, 5.76, 1.80, 1.81, 35.7)가 있는데 조건 토큰이 없음, 또는 금지 수치(MLA 문장의 42.5, ITME 문장의 3.02) |
| SX4 수치 | 같은 `metric_tag`·같은 기술의 수치가 10% 넘게 차이 |
| SX5 시점 | 반대 방향(positive vs negative) record 의 최신 근거 날짜가 12개월(365일) 넘게 차이 |
| SX6 stance | 같은 content_hash 근거가 support 와 counter 로 동시에 쓰임 |
| SX7 TRL 입력 | 시장·이해관계자 record 에 `pilot` 이상 근거가 있는데 `technical_findings.input_evidence_ids`에 없음 |

SW/HW 대조표(`contrast_table`)는 관점·기준별로 두 기술의 판단을 나란히 놓습니다. 두 기술은 다른 레이어라 기본 관계는 `independent`, 한쪽 record 가 없으면 `unknown`, 병행 주장이 있는 관점은 `complement`입니다.

### ③ write: 서술

LLM 에는 **근거 원문 전체가 아니라** 매트릭스, 관계 목록, record 요약, 인용 quote 와 메타데이터(제목·날짜·증거 수준·direct/proxy·stance)만 넘깁니다(§8.6).

| 서술기 | 용도 | 동작 |
|---|---|---|
| `TemplateWriter` (기본) | 오프라인 테스트, fixture 실행 | 관계를 종류·규칙·기술별로 묶어 정해진 문장으로 옮김. TRL 범위 문장 추가. 기술군·전망 근거는 자동 표기 |
| `OpenAIWriter` | 실제 서술 | OpenAI Responses API 구조화 출력. 요약 주장 5~10개 + 상충별 원인 설명. 기본 모델 `gpt-5.5`(환경변수 `SYNTHESIS_MODEL`로 변경), `max_output_tokens=16000` |

`OpenAIWriter`는 `max_output_tokens`를 넉넉히 잡았습니다. 이해관계자 에이전트에서 추론 토큰 때문에 응답이 잘린 문제(ISSUE 3-1a)를 겪었기 때문입니다. 응답이 `completed`가 아니면 상세 사유와 함께 `limitations`에 남기고 계속 진행합니다.

### ④ review: 중립성 검토 (C1~C7)

| 검사 | 내용 |
|---|---|
| 길이 | 한 문장 240자 이내 |
| C1 | `evidence_ids`가 1개 이상이고 모두 `evidence_store`에 있음 |
| C2 | `evidence_ids`가 네 관점이 실제 인용한 근거 안에 있음 |
| C3 | 문장 속 숫자가 인용 quote 또는 그 근거를 쓴 record 의 findings·value 에 있음 (제품명 속 숫자 V2·H800·67B, ID 속 숫자는 제외) |
| C4 | 조건 필수 수치의 조건 토큰 동반, 기술별 금지 수치 |
| C5 | 우열 어휘(우수·더 낫·승자·추천·권장·압도·우위·열위·월등) |
| C6 | 온디바이스·엣지·모바일 없음. proxy 근거 인용 시 "기술군", forecast 근거 인용 시 "전망" 표기 |
| C7 | 문장의 TRL 범위가 매트릭스 값과 같음. basis 가 direct 가 아닌 판정을 "직접 근거"로 쓰지 않음 |

위반 문장은 서술기의 `revise()`로 **한 번만** 다시 쓰고, 그래도 위반이면 제거해서 `dropped_sentences`에 사유와 함께 남깁니다. 상충 설명(`explanation`)에 우열 어휘가 있으면 설명만 지우고 `unresolved`로 둡니다.

검사 결과는 `meta.quality`에 들어갑니다: `passed`(제거 없음) / `needs_review`(일부 제거) / `failed`(남은 문장 없음).

---

## 5. 실행과 테스트

모든 명령은 레포 루트에서 실행합니다. 환경 설정은 루트 `README.md`의 "0. 환경 설정"을 따릅니다. (A) 가벼운 설치로 충분합니다.

### API 키 없이 (비용 없음)

```bash
# 테스트 36개
python -m unittest tests.agents.synthesis.test_synthesis -v

# 합성 fixture 로 전체 흐름 실행 (템플릿 서술)
python -m scripts.run_synthesis
```

`outputs/synthesis/<실행시각>/`에 두 파일이 생깁니다.

| 파일 | 내용 |
|---|---|
| `synthesis.json` | `SynthesisResult` 전체 (부모 State 에 들어갈 값) |
| `synthesis_report.md` | 사람이 읽는 보고서: 요약 주장, 매트릭스, 관계, 대조표, 공백, 재실행 요청, 한계, 제거된 문장 |

fixture 기준 결과: 매트릭스 12칸, 상충 9건(SX1~SX7 모두 탐지), 공유 근거 3건, 보완 3건, 요약 주장 14개 모두 검사 통과, 상태 `partial`(이해관계자·도메인 결과가 partial).

### 실제 LLM 서술 (키 필요, 비용 발생)

```bash
python -m scripts.run_synthesis --writer openai                 # .env 의 OPENAI_API_KEY 사용
python -m scripts.run_synthesis --writer openai --ask-key       # 키를 직접 입력
python -m scripts.run_synthesis --writer openai --model <모델ID>
```

이 스크립트는 이해관계자 스크립트와 달리 `.env`의 `OPENAI_API_KEY`도 읽습니다. 실제 LLM 서술은 아직 실행해 보지 않았고, 가짜 client 로 요청·응답 처리만 테스트했습니다.

### 다른 에이전트 결과로 실행

```bash
python -m scripts.run_synthesis --input <AppState.json>
```

`AppState` 형식(`graph/state.py`)의 JSON 이면 됩니다. 현재 도메인·이해관계자 에이전트는 아직 `PerspectiveFindings` 형식으로 결과를 내지 않으므로(ISSUE 1-2), 지금은 합성 fixture 로만 끝까지 확인할 수 있습니다.

### 부모 그래프에 연결

```python
from agents.synthesis import make_node
from agents.synthesis.writer import OpenAIWriter

build_graph(..., synthesis=make_node(OpenAIWriter()), ...)   # 실제 서술
build_graph(..., synthesis=make_node(), ...)                 # 템플릿 서술 (오프라인)
```

### 테스트가 확인하는 것

| 묶음 | 내용 |
|---|---|
| 노드 계약 | `synthesis` 키만 반환, 내부 필드 비노출, 필요한 키만 읽음, 전 관점 누락 시 예외 없이 `failed`, 서술 실패 기록, 부모 그래프 안에서 동작 |
| 매트릭스 | record 당 셀, content_hash 중복 제거, 누락 관점 → gap·재실행 요청, 없는 근거 ID 제외, not_applicable 제외, 2배 불균형, 기준일 이후 근거 |
| 관계 | fixture 에서 SX1~SX7 모두 탐지, 공유 근거는 일치가 아님, 어휘가 다르면 보완·같으면 일치, SX1 은 TRL 7 미만일 때만, unknown 은 SX2 제외, **sw/hw·입력 순서를 바꿔도 결과 동일**, 대조표 관계 |
| 검토 | C1·C3·C4·C5·C6·C7 위반 탐지, ID·제품명 숫자 제외, 1회 재생성 후 유지 / 재생성 후에도 위반이면 제거, 설명의 해소 상태, 템플릿 출력은 전부 통과 |
| OpenAIWriter | 가짜 client 로 quote 만 전달하는지, 응답 미완료 시 예외 대신 기록하는지 |

---

## 6. 합성 fixture

`tests/agents/synthesis/fixtures/appstate_sample.json`은 **테스트용 가짜 입력**입니다.
D1(DeepSeek-V2)·D3(ITME) 인용문은 설계서 §5에 적힌 원문 문장을 썼지만, 나머지 URL(example.com)과 시장 수치는 지어낸 값입니다. 파일 첫 필드 `_note`와 보고서 첫머리에 이 사실을 표시합니다.

모든 상충 규칙이 한 번씩 걸리도록 일부러 넣은 사례:
- SX1: ITME TRL 4-5 vs 시장 CXL 모듈 출시 발표(announcement)
- SX2: 시장의 CXL 기술군 전망 vs 도메인의 ITME 직접 판정
- SX3: "ITME는 처리량을 35.7% 높였다" (CPU-offload·최대 누락)
- SX4: 같은 `cxl_market_size_usd`에 15.8 / 3.4 billion USD
- SX5: 2024-05 시장 전망 vs 2026-08 도메인 반대 근거
- SX6: 같은 vLLM 문서가 시장에서는 support, 이해관계자에서는 counter
- SX7: TRL 입력에 없던 DeepSeek API 운영(production) 근거

---

## 7. 설계서와 다르게 해석한 부분

설계서에 조건이 정량적으로 정해지지 않은 부분은 아래처럼 정했습니다. 팀 논의로 바꿀 수 있습니다.

| 항목 | 설계서 | 구현한 해석 |
|---|---|---|
| record 의 "방향" | 명시 없음 | `stance_counts`의 support vs counter 다수 |
| SX1 "TRL 이 운영 환경 근거를 missing 으로 둠" | `TRLRecord.missing_evidence` 기준 | AppState 에 `TRLRecord`가 없어 **TRL 값 상한 < 7**로 판단 |
| SX2 "같은 주제" | 명시 없음 | 같은 기술 안의 class record 와 방향이 다른 direct record |
| SX5 "반대 방향 결론" | 명시 없음 | positive vs negative record, 각 record 의 최신 근거 날짜 비교 |
| SX7 "TRL 판정 이후 추가된 근거" | 명시 없음 | pilot 이상이면서 TRL 입력 목록에 없는 근거 |
| 판단 보류 항목 요약 | SUMMARY 에 포함 | 요약 주장은 근거 ID 가 필수(C1)라, 근거 없는 판단 보류 칸은 `gaps`와 매트릭스에만 남김 |
| LLM 상충 탐지 (`detected_by="llm"`) | 스키마에 있음 | 아직 규칙 탐지만 구현 |

---

## 8. State 에 제안할 것 (graph/state.py, 팀 리뷰 필요)

`graph/`는 공유 영역이라 직접 고치지 않았습니다.

1. **`Claim.perspective`가 관점 하나만 표현함.** 종합의 요약 주장은 여러 관점에 걸치는데 `Perspective`에 `synthesis`가 없습니다. 지금은 인용 근거의 첫 관점을 넣습니다. `perspective: Perspective | Literal["synthesis"]`를 제안합니다.
2. **`TRLRecord`를 담을 자리가 없음.** 설계서 §6.2.3의 `supporting_evidence` / `blocking_evidence` / `missing_evidence`가 있어야 SX1 을 설계대로 판정할 수 있습니다.
3. **보고서 초안.** 설계서 §7.2 는 ⑤가 `report_sections`에 SUMMARY·시사점·한계 초안을 쓴다고 되어 있습니다. 이번에는 `synthesis`만 만들었고, 초안은 ⑥ 담당과 형식을 정한 뒤 추가합니다.
4. **품질 결과 위치.** 중립성 검사 결과를 `synthesis.meta.quality`에 두었습니다. `quality_by_perspective["synthesis"]`에도 쓸지 정해야 합니다.
