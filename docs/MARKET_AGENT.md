# 시장 평가 에이전트 설계

KV cache 최적화 기술 두 건(SW: DeepSeek-V2 MLA, HW: ITME CXL-Hybrid)이 **데이터센터** LLM
추론(서빙) 시장에서 어떤 채택·생태계 신호를 갖는지 조사하는 에이전트다. 우열을 판정하지
않고, 근거의 확인 정도(basis: direct/inferred/unknown)와 관점(support/counter)을 함께
기록한다.

## 책임 경계 (설계서 §2.1 — 하지 않는 일)

| 규칙 | 어디서 지키는가 |
| --- | --- |
| 데이터센터 서빙 밖(온디바이스·엣지·모바일·학습 전용) 근거를 채택하지 않는다 | `agents/market/prompts.py`의 `JUDGE_SYSTEM` 규칙 2 — LLM 판단 단계에서 `relevant=false` 처리 |
| 논문 그대로를 production(실운용)으로 세지 않는다 | `agents/market/subgraph.py`의 `organize()` — `level = "pilot" if (is_paper and ...production) else ...` |
| 다른 관점(이해관계자·도메인)의 결론을 입력받지 않는다 | `agents/market/node.py`의 `project_input()` — `technical_findings` 요약만 통과시키고 그 외 부모 State 키는 보지 않는다(`test_project_input은_다른_관점_결과를_건드리지_않는다`로 검증) |
| 시장 채택 지표를 TRL 등 다른 관점의 성숙도 판정과 혼동하지 않는다 | `quality/rubric.py`는 시장 신호(`market_signal`: adopted/announced/projected/none)만 계산하고, TRL 단계 판정은 만들지 않는다 |

이 표에 없는 새 "하지 않는 일"을 코드로 추가할 때도 표를 같이 갱신한다.

## 검색 경로: 하이브리드

시장 규모·성장성은 웹 검색만 쓴다. 리포트·뉴스 수치는 정적 인덱스로 고정하면 시점이
왜곡되기 때문이다. 반대로 상용화·채택, 생태계 지지, 반대·한계 근거 세 칸은 웹 검색과
**Pool B(런타임 수집 코퍼스)**를 함께 쓴다.

## Pool B — 런타임 수집 코퍼스 (설계서 §3.1 다이어그램)

```
평가 질문 생성 → 웹 검색(agents/market/rag/tools.py)
             → 출처 등급 필터(agents/market/rag/tier.py, §1.4)
             → 본문 수집(agents/market/rag/fetch.py, 페이지 예산 검사)
             → 문서 길이 판단
                 짧음(≤ SHORT_DOCUMENT_CHARS) → 그대로 근거 후보
                 김                            → 청킹 + BM25 색인(agents/market/rag/index.py)
                                                  → 질의에 맞는 조각만 근거 후보
```

- **출처 등급 필터** (`tier.py`): 검색 API가 준 도메인을 그대로 믿지 않는다. `paper`/
  `patent`/`standard`/`vendor`/`news` 등급표에 없는 도메인("기타")은 채택하지 않고 이유를
  `search_log_by_perspective["market"]`에 남긴다. 등급표는 하드코딩된 화이트리스트라
  완전하지 않다 — 실제 실행에서 정당한 출처(예: 증권사 리서치)가 걸러지는 사례가
  나오면 `tier.py`의 등급표에 추가한다.
- **본문 수집** (`fetch.py`): 검색 스니펫만으로는 배포 조건·검증 환경·한계를 판단하기
  어렵다. `PAGE_BUDGET`(기본 200p, `CHARS_PER_PAGE=3000` 환산)을 누적 검사하며, 한도
  초과분은 건너뛰고 이유를 로그에 남긴다.
- **길이 판단**: `SHORT_DOCUMENT_CHARS`(8000자) 이하면 그대로 근거 후보로 쓴다. 넘으면
  `index.py`가 `CHUNK_CHARS=1200`/`CHUNK_OVERLAP=200`으로 청킹하고, BM25로 검색 질의와
  가장 관련 있는 조각 최대 3개만 근거 후보로 올린다(전체 본문을 LLM에 넣지 않는다).
- `MarketAgentDeps.fetch_body=None`이면(오프라인 테스트 기본값) 이 단계를 건너뛰고 예전처럼
  검색 스니펫만 쓴다 — 기존 테스트를 깨지 않기 위한 선택 기능이다.

## 판정은 코드가 한다 (agents/market/quality/rubric.py)

기술 × 세부 기준(시장 규모·채택·생태계) 6칸마다 "충분/부분/부족"을 규칙으로 계산한다.
LLM에게 판정을 맡기면 같은 근거를 넣어도 실행마다 답이 달라진다. 예:

- 시장 규모: 서로 다른 출처 2건 이상 + 수치가 있어야 충분
- 상용화·채택: **논문·시뮬레이션·프로토타입은 실운용 근거로 인정하지 않는다.** 공식·언론
  출처의 실서비스 증거가 1건 이상이어야 충분
- 블로그·커뮤니티 근거만으로는 조건을 만족해도 "충분"으로 올리지 않는다
- 기술마다 반대(counter) 근거가 최소 1건 없으면 그 기술은 미완료로 남는다

## 근거 검증 2단계

1. **인용 원문 검증(코드)**: LLM이 "이 문장이 근거"라고 지목한 인용문이 실제 검색 결과
   본문에 있는지 문자열 대조로 확인한다. 없으면(환각) 통째로 버린다.
2. **반대 근거 재검증(LLM, 별도 호출)**: `stance=counter`로 분류된 근거만 다시 검증한다.
   ITME 논문 초록을 실제로 넣어 봤더니, "기존 DPU 오프로딩 방식의 소프트웨어 최적화·비용
   부담"이라는 **비교 대상의 한계**를 ITME 자신의 한계로 옮겨 적는 사례를 발견했다. 재검증은
   인용문만 다시 보여주고 "이 한계의 주어가 대상 기술 자신인가"만 판정한다(요약문은 주지
   않는다 — 요약문 자체가 이미 잘못 귀속됐을 수 있어서 원문 인용만 신뢰한다).

## 근거 ID (agents/market/rag/evidence.py)

도메인 에이전트와 같은 방식으로, URL 정규화 + 위치 + 인용문에서 해시로 ID를 유도한다
(`market:ev:<hash12>`). 같은 자료를 재시도로 다시 가져와도 같은 ID가 나온다.

## 부모 State: graph/state.py의 AppState (ISSUE.md 1-1 해결됨)

2026-09-22 "Replace parent State with team AppState" 커밋으로 부모 State가 `graph/state.py`의
`AppState` 하나로 확정됐다(`graph/team_state.py`는 삭제됨). `make_node()`는 더 이상
`legacy` 분기를 받지 않는다.

```python
from agents.market import make_node, MarketAgentDeps

deps = MarketAgentDeps(llm=..., strong_llm=..., web_search=..., retriever=..., fetch_body=...)
node = make_node(deps)
```

내부에서는 `stance`/`scope`/`observed_at`을 `Claim`의 실제 필드로 다루고(`subgraph.py`는
바뀌지 않았다), `agents/market/state.py`의 `to_perspective_findings()`가 노드 경계에서
`graph.state.PerspectiveFindings`/`Evidence`/`Claim`/`Gap`/`VerdictRecord` 형식으로 바꾼다.

기존 `충분/부분/부족` 판정은 설계서 §2.2.4 표대로 `basis`(`direct`/`inferred`/`unknown`)에
그대로 대응된다(`quality/rubric.py`는 바꾸지 않았다). `assessment`(market_signal 어휘:
adopted/announced/projected/none) 같은 일부 매핑은 설계서에 표가 없어
`agents/market/state.py`의 `to_perspective_findings` 근처 주석에 직접 정한 근거를 적어
뒀다.

## 입력 / 출력

| | 내용 |
| --- | --- |
| 입력 | `selected_tech.sw/hw`, `domain`, `request.as_of`, `request.max_search_rounds`, 선행 `technical_findings` 요약(다른 관점 결과는 받지 않는다) |
| 출력 | `market_findings`(`PerspectiveFindings`: records/claims/gaps/limitations/input_evidence_ids), `evidence_store`(`Evidence`, id 키), `search_log_by_perspective`(관점별 검색·필터 로그) |

## 런타임 의존성 (make_node 인자로 주입, 전역 상태 없음)

```python
@dataclass
class MarketAgentDeps:
    llm: object            # 계획·판정용. .with_structured_output(...) 지원 필요
    strong_llm: object | None = None   # 반대 근거 재검증·비교 추론용. None이면 llm 사용
    web_search: Callable[[str], list[SearchResult]]
    retriever: Callable[[str, str | None], list[SearchResult]] | None = None
    fetch_body: Callable[[str], FetchedDocument] | None = None  # Pool B 본문 수집. None이면 스니펫만
    page_budget: int = 200
```

`agents/market/rag/tools.py`에 `stub_web_search`(오프라인용)와 `tavily_web_search`가 있다.
공유 RAG 검색기가 나오면 `agents/market/rag/retriever.py`의 `make_rag_fn(retriever)`로
감싸서 `deps.retriever`에 넣으면 된다. 아직 팀 공유 검색기가 없어 `fake_retriever`(같은
파일)로 어댑터 계약만 테스트했다.

## 완료 기준

6칸(기술×기준) 모두 부족이 아니고 기술마다 반대 근거가 있으면 `complete`. 부족한 칸이나
반대 근거 누락이 남으면 그 목록을 `gaps`에 적고 `partial`. 근거를 하나도 판정하지 못했으면
(검색·판단 자체가 전부 실패) `failed` — 개별 근거가 품질 기준으로 거절된 것은 `failed`가
아니라 `partial`이다(둘을 섞으면 "정상적으로 다 걸러냈다"와 "아예 못 돌았다"를 구분할 수
없다).

## 단독 실행 (python -m scripts.run_market)

```bash
cp .env.example .env   # OPENAI_API_KEY, TAVILY_API_KEY 입력
python -m scripts.run_market --rounds 1        # 가장 작은 규모, API 비용 발생
python -m scripts.run_market --offline         # API 키·네트워크 없이 배선만 확인(스텁)
```

결과는 `outputs/market/<실행시각>/`에 저장된다 — `market_findings.json`,
`evidence_store.json`, `search_log.json`(Pool B 필터·예산 로그), `market_report.md`.
본문 수집 캐시는 산출물이 아니라 `data/fetch_cache/market/`에 저장된다(`.gitignore`에서
`data/fetch_cache/`로 제외됨).

## 테스트

```bash
python tests/agents/market/test_market_agent.py   # 또는 pytest tests/agents/market/
```

API 키·네트워크 없이 27개 테스트가 돈다: 근거 ID 결정성 3종, Rubric 판정 6종, 표현 린터,
출처 등급 필터 2종, Pool B 본문 수집·청킹·페이지 예산 4종, AppState 변환(`to_perspective_findings`)
5종, 가짜 LLM으로 만든 전체 흐름(정상/환각 기각/반대 근거 오귀속 차단/RAG 라우팅/입력 격리) 5종,
판단 프롬프트 회귀 검사 1종.

## 구현 상태와 한계

- **실제 API로 end-to-end 실행 4회 확인함**(2026-09-22). 매번 같은 패턴이 재현됐다: "시장
  규모"는 출처 등급 필터를 거치지 않아 잘 채워지지만, "상용화·채택/생태계 지지/반대근거"는
  필터를 거쳐서 등급표가 좁을수록 대부분 비었다. `report (2).pdf`(등급표 확장 전 실행)는
  8칸(기술 2 × 기준 4) 전부 `not_found`로 나온 사례였다.
- **등급표 확장(`rag/tier.py`)**: 위 문제의 원인을 `search_log.json`으로 직접 확인했다 — 실제로
  `api-docs.deepseek.com`(대상 SW 원저작사!), `semanticscholar.org`(대상 HW 논문 원문 링크),
  `alphaxiv.org`, `databricks.com`, `redhat.com`, `mordorintelligence.com` 같은 정당한 출처가
  등급표에 없어 걸러지고 있었다. 이들을 추가하고(`vendor`/`paper`/신설 `research` 등급) 재실행해
  개선을 확인했다 — 등급표 확장 전 8칸 중 0~1칸이던 것이 확장 후 2~4칸으로 늘었다. 다만
  상용화·채택/생태계는 여전히 자주 비어서 등급표를 더 넓히거나 관점별 검색 질의를 다양화할
  필요가 남아 있다.
- **근거 오귀속 방지(`prompts.py` JUDGE_SYSTEM 규칙 2)**: 실행 중 SK hynix의 별개 제품
  "IMTE"(Inference **M**emory **T**iering **E**xpansion)를 대상 HW 기술 "ITME"(Inference
  **T**iered **M**emory Expansion)로 오귀속할 뻔한 사례를 발견했다 — 약어 철자 순서만 다르고
  개념도 비슷해 LLM이 혼동하기 쉽다. "발행 주체가 다른 별개 기술은 relevant=false" 규칙을
  추가했다. LLM이 실제로 이 규칙을 지키는지는 실 API로만 확인 가능하므로, 오프라인 테스트는
  규칙 문구가 프롬프트에서 빠지지 않는지만 회귀 검사한다.
- 질의 계획을 매 실행 LLM이 새로 만들어서, 검색되는 자료와 결과가 실행마다 달라진다.
  안정화하려면 질의를 캐시하거나 `seed_urls`로 신뢰 출처를 지정하는 방법이 있다.
- 모델 구조(MLA)처럼 자체 시장이 없는 기술은 "시장 규모" 칸이 계속 부족으로 남을 수 있다.
  Rubric에 "적용 불가(not_applicable)" 판정을 둘지는 팀 결정 사항이다.
- LLM 판단 오류 가능성은 원문 인용 검증과 반대 근거 재검증으로 줄였지만 제거하지는 못한다.
- `input_evidence_ids`(TRL 입력 불일치 검사용)는 항상 빈 배열이다 — 시장 에이전트가
  `technical_findings`를 요약해 프롬프트에는 넣지만, 그 근거 ID를 추적해 넘기지는 않는다.
