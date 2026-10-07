# Subject

본 프로젝트는 KV cache 최적화 기술을 소프트웨어, 하드웨어 두 진영에서 선정하여,
기술 성숙도·시장·이해관계자·도메인 관점에서 평가하는 **Supervisor 패턴** 기반 멀티 에이전트를 설계/개발하는 프로젝트 임.

우열을 매기는 것이 목적이 아니라 "어떤 조건에서 적합한가"를 근거와 함께 남기는 것이 목적이다.
RAG 과제 단계의 상세 설명(검색 실험, 에이전트별 판정 규칙 등)은 [docs/RAG_README.md](docs/RAG_README.md)에 있다.

## Overview

- **Objective** : 하나의 기술을 복수 관점에서 비교 평가. 관점별 판정과 그 판정이 성립하는 조건, 판단하지 못한 공백(gap)을 함께 기록
- **Pattern** : **Supervisor** — 관점 에이전트의 결과 품질(근거 충분도)이 실행마다 달라, 결과를 점검한 뒤 다음 행동(재조사·종합·재작성)을 정하는 중앙 판단이 필요했다.
  이 프로젝트에서는 다음 행동의 판단과 재작업 예산 관리를 supervisor에 모았다.
- **동적 처리** : `technical → 3관점 → synthesis → report → quality_eval`의 선행 의존성은 유지한다. 각 워커는 supervisor로 돌아오며, supervisor가 매 단계 State를 읽어
  - 근거가 부족한 관점만 재작업시키고(충분한 관점은 그대로 통과),
  - 관점 결과가 바뀌면 synthesis·report를 stale로 보고 다시 실행하며,
  - 품질 평가가 미달이면 원인 관점(잘못된 인용의 출처 에이전트) 재작업 또는 피드백을 반영한 보고서 재작성을 고른다.
  - 저장된 live 실행 `20261007-171209-920622`에서는 market·domain 재작업 후 보고서를 생성하고, 품질 평가가 지목한 technical을 재작업한 뒤 synthesis·report·quality_eval을 다시 실행했다.
  - 품질 미달이어도 허용된 종료 선택이나 실행 상한으로 끝날 수 있다. 종료와 품질 통과는 별도로 확인한다.

## Selected Technologies

- **SW : DeepSeek-V2 Multi-head Latent Attention (MLA)** (arXiv:2405.04434) — 어텐션 구조에서 KV를 저차원 잠재 표현으로 압축해 KV cache 총량을 줄인다.
  양자화·축출과 달리 모델 구조를 바꾸는 SW 대표 사례이며, 원논문이 측정 조건과 함께 정량 수치를 공개해 근거 검증이 가능하다.
- **HW : ITME (Inference Tiered Memory Expansion)** (arXiv:2606.12556) — CXL-hybrid 계층 메모리로 KV 저장 공간을 GPU HBM 밖으로 확장한다.
  SW가 "KV를 줄인다"면 HW는 "KV를 둘 자리를 늘린다" — 같은 병목을 반대 방향에서 다뤄 비교 대상으로 삼았다.

## Features

- **PDF 자료 기반 정보 추출** — Pool A 고정 코퍼스(arXiv 6편, 136p)를 SHA-256 매니페스트로 고정, BM25+Dense+RRF 하이브리드 검색
- **웹 근거 수집** — Tavily(기술·시장·도메인), OpenAI `web_search` + 원문 HTML 대조(이해관계자)
- **근거-주장 결합 검증** — 이해관계자 인용문은 원문 block·해시·날짜·수치를 코드로 대조. 도메인은 claim 본문과 record의 value에 있는 측정값을 인용 quote와 대조해, 미확인 claim은 제외하고 record의 value는 비운다. 의미·조건·인과 귀속 전체의 일치까지 보장하는 검사는 아니다.
- **확증 편향 방지 전략**
  - 관점 격리: 각 관점은 다른 관점의 결론을 입력으로 받지 않는다(공통 입력은 기술 조사 요약뿐)
  - 기대 효과·제약을 함께 조사하고 반대 근거(counter)를 확인. 반대 근거 부재는 품질 검사에서 경고로 남긴다
  - 섹션 분량 선별 시 SW·HW를 번갈아 고르고, 입력에 반대 근거가 있으면 기술마다 1건을 우선 선택
  - 근거 부재는 "없다"가 아니라 "공개 근거에서 확인하지 못했다"로 기록
- **보고서 품질 평가 (Hybrid, 3안)** — 실험으로 역할을 나눴다([docs/QUALITY_EVAL.md](docs/QUALITY_EVAL.md))

  | 평가 항목 | 코드 검사 | LLM Judge (gpt-4o) |
  |---|---|---|
  | Groundedness | 평가 대상 사실 불릿·표 행 인용률 ≥ 0.8, 미존재 인용 ID·무인용 측정값·보고서 차단 위반 0건 | 인용된 측정값 행을 앞에서 최대 15개, 행당 인용 근거 최대 2개·각 약 400자로 대조. 점수 ≥ 3 AND unsupported 0건 |
  | 중립성 | 우열·추천 금지어 | 금지어 없는 암묵적 추천 탐지 |
  | 편향 통제 | 전체 최대 출처 비중 ≤ 0.4, SW/HW 인용 비율 ≤ 2.5. 인용 행 ≥ 3개인 관점 섹션은 단일 출처 > 0.6 경고·> 0.8 실패. 한 기술만 다루면 실패하되 상대 기술 근거 부재가 공백으로 명시되면 경고 | — (보고서 통째 채점 실험에서 4개 모델 모두 탐지 실패) |
  | 관점 커버리지 | 4.1~4.4 섹션 존재·인용·실질 내용 | — (동일) |

  Groundedness 분모는 `graph/groundedness.py`가 정한 사실 불릿·표 행이며, 기술 선정·남은 확인 과제·한계점과 일부 메타 문구는 제외한다. 일반 산문 전체를 검사하는 방식은 아니다.
  미달이면 supervisor는 관점 재작업·보고서 재작성·미달 종료 중 허용된 행동을 선택한다(보고서 최대 2버전). 보고서는 품질 피드백을 입력으로 받으며, 관점별 전달 범위는 아래 현재 한계 참고.
- **보고서 10장 제한** — 섹션별 분량 예산(초과는 경고) + 제출본 인용 번호화(`[n]`, 코드가 REFERENCE 순서로 계산) + PDF 장수 가드. `--no-pdf` 실행은 추정 장수를 사용한다.

## Tech Stack

- **Framework** : LangGraph (`requirements.txt`: ≥ 1.0.1, < 2; lock: 1.2.11, StateGraph·conditional edges·MemorySaver), LangChain (`requirements.txt`: ≥ 1.0.2), LangSmith tracing
- **LLM/Supervisor** : gpt-4.1 (규칙이 계산한 허용 행동 안에서만 선택, 위반 시 규칙 기본값으로 대체)
- **LLM/Generator** : 에이전트별 — 기술 조사 gpt-4.1, 시장 gpt-4.1-mini(+gpt-4.1 재검증), 이해관계자 gpt-5-mini, 도메인 gpt-4o, 평가 종합 gpt-4.1, 보고서 gpt-4.1 (gpt-4o-mini·gpt-5.1과 비교해 지어낸 인용·속도·분량 기준으로 선정)
- **LLM/Judge** : 기본 gpt-4o (`QUALITY_JUDGE_MODEL`로 변경 가능). 단일 기준 보고서의 5종 변형을 4개 모델로 각각 2회 평가한 v2 실험에서 gpt-4o의 탐지율·반복 일관성은 100%, 기록된 false positive는 50%. 이 결과는 합성 변형 실험이며 실제 보고서의 정확도를 보장하지 않는다([실험 결과](outputs/quality/judge_experiment_v2/results.md), [해석·한계](docs/QUALITY_EVAL.md)).
- **Retrieval** : BM25 + FAISS(Dense) + RRF — 도메인 검색 ablation의 Hit@5 0.875, MRR 0.833 (Golden Set 16문항, 청크 213개; [결과](outputs/ablation.json)). 전체 에이전트 공통 성능 지표는 아니다.
- **Embedding** : BAAI/bge-m3 (오픈소스, 8192 토큰, 교차언어 검색)

## Agents

- **Supervisor** (`graph/supervisor.py`) : 규칙이 State에서 허용 결정 집합(예산·상한·의존성 반영)과 기본값을 계산 → LLM이 그 안에서 선택 → guard가 검증.
  `status=complete`는 충분으로 인정한다. 그 외 결과는 실재 인용 근거 ≥ 3건 AND record의 direct/inferred 비율 ≥ 0.5를 적용하며, 결과 없음·failed는 부족으로 판정한다. 충분하거나 부족 상태가 수용된 뒤 종합·보고서로 진행한다.
- **① 기술 조사** (`agents/technical`) : Pool A RAG + Tavily, TRL 1~9 Gate는 코드가 계산
- **② 시장 평가** (`agents/market`) : 시장 규모·상용화·생태계, 출처 등급 필터, 반대 근거 재귀속 검증
- **③ 이해관계자 평가** (`agents/stakeholder`) : 기술 × 그룹 발언을 원문 대조 후 지지·반대·중립 분류
- **④ 도메인 평가** (`agents/domain`) : 데이터센터 요구사항 6축 × 기술 2개 판정
- **⑤ 평가 종합** (`agents/synthesis`) : 비교 매트릭스·상충(SX1~SX7)·보완 관계, 중립성 검사(C1~C7)
- **⑥ 보고서 생성** (`agents/report`) : 서술은 LLM, TRL·비교표·조건표·인용·REFERENCE·핵심 수치 조건 처리는 코드. 입력 밖 인용 ID를 제거하고, 부분 수정 후 차단 위반이 남으면 검증을 통과한 결정적 렌더로 교체한다. 결정적 검사는 원문 외 상위 claim·record도 참조하므로 원문 함의 일치와는 구분한다.
- **⑦ 품질 평가** (`agents/quality`) : 위 Hybrid 평가, 근거가 뒷받침하지 못한 주장을 출처 관점 재작업으로 라우팅

## State Schema

`graph/state.py`의 `SupervisorState`(제어 메타) ⊃ `AppState`(관점 결과 계약). 에이전트 내부 State는 각 에이전트 서브그래프에 따로 있다(Layered).

- **제어 vs 페이로드 분리** : 제어 필드와 관점 결과를 같은 State 안에서 구분한다. supervisor는 실행 메타·충분도·결과 갱신 시점·synthesis 존재 여부·report_version·eval_result로 다음 행동을 계산하고 관점 결과는 수정하지 않는다. 분기 함수 `route()`는 계산된 `next`만 읽는다.
- **관측성 위치** : 결정 전체 로그는 State 밖 `outputs/graph/<run>/decisions.jsonl`(`{trace_id, step, decision, reason, source, ts}`). State엔 `last_decision` 1건만. LangSmith run metadata에 `trace_id`
- **지속성 비용** : CLI 워커는 완성 Markdown·PDF를 파일로 쓰고 `artifacts`에 경로를 둔다. 개별 보고서 섹션 본문·관점 결과·evidence quote·검색 로그는 State에 남아 체크포인트 비용을 발생시킨다. fetch_cache와 결정 JSONL은 State 밖에 두고, `last_decision`은 최신 1건으로 덮어쓴다.
- **상관** : `trace_id` = checkpoint `thread_id` = LangSmith metadata/tag = decisions.jsonl 키
- **재개/복구** : 전체 State(관점 결과·종합·품질 판정 포함)와 실행 메타를 바탕으로 다음 행동을 계산한다. `run_graph()`는 호출마다 `MemorySaver`를 생성하므로 체크포인트는 해당 실행의 메모리에 한정된다. 프로세스 재시작 후 복구나 CLI 재개 옵션은 구현하지 않았다. 워커 예외는 `failed` 상태로 기록해 후속 판단을 계속한다.
- **동시 처리** : 병렬 워커가 함께 쓰는 키는 reducer — `evidence_store`(ID 기준 멱등 병합), `node_status`·`artifacts`·`quality_by_perspective`·`search_log_by_perspective`·`run_meta`(dict 병합). 나머지는 노드별 소유
- **종료 보장** : supervisor 진입 시 `step_count ≥ max(0, max_steps - 1)`이면 종료만 허용하고 카운터를 1 증가시킨다(기본 max_steps=20). 관점당 재작업 1회, 보고서 최대 2버전이며 LLM 제안은 허용 행동으로 제한된다. 미달 종료는 품질 통과가 아니며 `eval_result.passed`와 종료 사유로 구분한다.

## Architecture

![Supervisor graph](outputs/architecture/supervisor_graph.png)

모든 워커의 유일한 출구는 supervisor(실선)이고, supervisor의 분기(점선)는 `add_conditional_edges`로 State에서 계산된다.
market·stakeholder·domain은 같은 superstep에 병렬로 디스패치될 수 있다.

## Directory Structure

```
├── agents/                 # 하위 에이전트 (관점 간 판단 입력 격리, 프롬프트는 각 폴더의 prompts.py)
│   ├── technical/          # ① 기술 조사
│   ├── market/             # ② 시장 평가
│   ├── stakeholder/        # ③ 이해관계자 평가 (원문 검증형)
│   ├── domain/             # ④ 도메인 평가
│   ├── synthesis/          # ⑤ 평가 종합
│   ├── report/             # ⑥ 보고서 생성 (분량 예산·수치 조건·PDF)
│   └── quality/            # ⑦ 품질 평가 (Hybrid)
├── graph/                  # 조정 계층
│   ├── state.py            # SupervisorState / AppState / reducer
│   ├── supervisor.py       # 허용 결정 계산 + LLM 선택 + guard
│   ├── workers.py          # 워커 래퍼 (재작업 지시 주입, 예외 격리, 산출물 파일화)
│   ├── build.py            # 그래프 토폴로지
│   ├── decision_log.py     # decisions.jsonl
│   ├── rules.py            # 공유 보고서 규칙 표 (금지어·핵심 수치 조건)
│   ├── groundedness.py     # 공유 근거 연결 기준 (사실 문장 분류·인용률 하한)
│   ├── metrics.py          # 공유 수치 추출
│   └── stubs.py            # 오프라인 fixture 재생 노드
├── data/                   # 문서 풀 (Pool A 고정 PDF, 캐시)
├── scripts/                # 에이전트 단독 실행, judge 실험
├── tests/                  # 오프라인 테스트 (API 키 불필요)
├── docs/                   # 설계 문서 (SUPERVISOR_PLAN, QUALITY_EVAL, 에이전트별)
├── outputs/                # 실행 결과, 아키텍처 그림, judge 실험 결과
├── main.py                 # 실행 스크립트
└── README.md
```

## Usage

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest         # 테스트 실행용; requirements.txt에는 미포함
cp .env.example .env        # OPENAI_API_KEY, TAVILY_API_KEY, LANGSMITH_API_KEY
```

```bash
python main.py                                   # 관점 fixture + 종합/보고서 deterministic + code-only 품질 검사
python main.py --live all --supervisor llm       # 실제 실행 + LangSmith 트레이싱
python -m pytest tests -q --ignore=tests/agents/report/test_report_llm_integration.py
```

결과는 `outputs/graph/<실행시각>/`에 `report.pdf`, `report.md`, `decisions.jsonl`, `summary.md`(라우팅 경로), `final_state.json`으로 남는다.

PDF 출력에는 ReportLab과 한글 TTF/TTC 폰트가 필요하다. 자동 탐색 경로에 폰트가 없으면 `REPORT_PDF_FONT`로 지정하고, PDF가 필요 없는 실행은 `--no-pdf`를 쓴다.
live 실행에서 `LANGSMITH_API_KEY`가 있으면 tracing을 기본 활성화한다. 기본 실행은 API·tracing 전송을 끈다.
`main.py`의 종료 코드 0은 debug trace에 노출된 task error가 없다는 뜻이다. 워커가 State로 기록한 실패나 품질 미달은 종료 코드만으로 구분할 수 없으므로 `node_status`, `eval_result`, `summary.md`를 함께 확인한다.

`outputs/graph/`는 Git 추적 대상에서 제외된다. 제출할 PDF와 `tracing-1.png`, `tracing-2.png` 등 LangSmith 캡처는 별도로 준비한다.

## Current Limitations

- 2026-10-07 로컬에 저장된 live 실행 6건은 모두 최종 품질 미달로 종료했다. 최신 `20261007-171209-920622`는 7쪽 PDF를 생성했지만 groundedness·bias_control이 미달이었다. 동적 반복의 실행 증거와 보고서 품질 통과 증거는 구분한다.
- 품질 재작업 시 보고서에는 피드백, 이해관계자에는 focus/gap을 전달한다. 시장·도메인은 검색 라운드 확대만 입력에 반영하고, 기술 조사는 고정 요청을 재구성한다. 기술·시장·도메인에 실패 문장·근거 ID를 전달해 수정하는 연결은 추가 구현이 필요하다.
- 원문 함의 검사에는 측정값 행·인용 개수·발췌 길이 제한이 있다. 전체 보고서의 모든 주장·조건·인과 관계가 검증됐다는 의미는 아니다.
- 현재 재현성은 저장된 실행 결과와 별개로 설치 환경에서 확인해야 한다. 2026-10-07 검토 환경의 기존 `.venv`에는 `langgraph`·`pytest`가 없어 테스트와 CLI 실행을 검증하지 못했다.

## Contributors

| 이름 | 수행 역할 |
|---|---|
| 강유성 | ① 기술 조사 에이전트(Pool A 코퍼스·하이브리드 검색·TRL Gate), Supervisor·State 설계, 보고서 분량 예산, 품질 평가 Hybrid 설계·judge 실험 |
| 이효은 | ② 시장 평가 에이전트(출처 등급 필터, rubric 판정, 반대 근거 재검증) |
| 지승환 | ③ 이해관계자 평가 에이전트(발언 수집, 편향 플래그), 아키텍처 다이어그램 |
| 이산 | ④ 도메인 평가 에이전트(웹 검색 RAG, 임베딩·검색 방식 비교 실험) |
| 안균승 | ⑤ 평가 종합 에이전트(비교 매트릭스, 상충 규칙 SX1~SX7, 중립성 검사) |
| 이동영 | ⑥ 보고서 생성 에이전트(섹션 서술, 인용·REFERENCE 검증, PDF 출력) |
