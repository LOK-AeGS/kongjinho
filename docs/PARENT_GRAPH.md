# 부모 그래프

설계서 v0.8 §8.1 전체 아키텍처를 지금 있는 노드로 연결한 것입니다.

```
START → ① technical → ② market ─┐
                     ③ stakeholder ├→ ⑤ synthesis → ⑥ report → END
                     ④ domain ─────┘
```

①이 먼저 실행되고, ②③④는 같은 단계에서 병렬로 실행되며 서로의 결과를 보지 않습니다.
세 결과가 모두 모인 뒤 ⑤가 실행되고, `evidence_store`는 이 지점에서 reducer(`merge_evidence_store`)로 합쳐집니다.

---

## 1. 파일

| 파일 | 역할 |
|---|---|
| `graph/build.py` | 엣지 정의 (위 흐름). 노드 함수는 밖에서 받음 |
| `graph/state.py` | 공통 State `AppState`, reducer, `create_initial_state` |
| `graph/stubs.py` | **임시 노드**: 아직 AppState 형식 노드가 없는 자리를 합성 fixture 재생으로 채움 |
| `main.py` | 실행 진입점. 노드 조립 → `build_graph` → 실행 → 결과 저장 |
| `tests/graph/test_parent_graph.py` | 부모 그래프 테스트 11개 (API·네트워크 없음) |

규칙대로 엣지는 `graph/build.py`에만 있고, `main.py`는 각 에이전트의 `make_node()`만 가져다 씁니다.

---

## 2. 노드별 연결 상태 (2026-09-22)

| 노드 | 기본 실행 (`python main.py`) | `--live` 실행 | 비고 |
|---|---|---|---|
| ① technical | 임시 (fixture 재생) | 불가 | 기술 조사 에이전트 PR 전 |
| ② market | fixture 재생 | `agents.market` (gpt-4.1-mini/gpt-4.1 + Tavily) | AppState 형식 ✅ |
| ③ stakeholder | fixture 재생 | `agents.stakeholder_eval` (gpt-4.1-mini, OpenAI 웹 검색) | AppState 형식 ✅. 기존 `agents/stakeholder/`는 옛 형식 |
| ④ domain | fixture 재생 | `agents.domain` (gpt-4o + Tavily, 긴 문서는 bge-m3 임베딩) | AppState 형식 ✅ (PR #8). 모델·임베딩은 환경변수 `DOMAIN_MODEL`, `DOMAIN_EMBEDDING`(빈 값이면 BM25만) |
| ⑤ synthesis | 실제 노드, 템플릿 서술 | `agents.synthesis` + gpt-4.1 서술 | AppState 형식 ✅ |
| ⑥ report | 실제 노드, deterministic | `agents.report` (gpt-4o-mini) | AppState 형식 ✅ |

fixture 는 `tests/agents/synthesis/fixtures/appstate_sample.json`(합성 자료)입니다. 재생 노드는 그 관점의 결과와, 그 관점이 인용한 근거만 `evidence_store`로 돌려줍니다.
D1·D3 인용 외의 URL·수치는 가짜라서, 재생 노드가 하나라도 있으면 결과에 "실제 조사가 아님" 경고가 붙습니다.

**실제 노드로 바꾸는 법.** `main.py`의 `build_nodes()`에서 해당 줄을 그 에이전트의 `make_node(...)`로 바꾸면 됩니다.
다 바뀌면 `graph/stubs.py`는 지웁니다.

---

## 3. 실행

```bash
# 전부 오프라인 (API 키 불필요, 비용 없음)
python main.py

# 일부만 실제 실행 (.env 의 OPENAI_API_KEY 사용, 비용 발생)
python main.py --live synthesis
python main.py --live synthesis,report

# 가능한 노드 전부 실제 실행 (market·domain 은 TAVILY_API_KEY 도 필요)
python main.py --live market,stakeholder,domain,synthesis,report --rounds 1
```

| 옵션 | 내용 |
|---|---|
| `--live` | 실제로 실행할 노드 (쉼표 구분): `market`, `stakeholder`, `domain`, `synthesis`, `report` |
| `--rounds` | 실제 실행 노드의 최대 검색 라운드 (기본 1, `request.max_search_rounds`로 들어감) |
| `--as-of` | 조사 기준일 (기본: fixture 의 기준일) |
| `--fixture` | 재생 노드가 쓸 AppState JSON |
| `--output-dir` | 결과 폴더 (기본 `outputs/graph`) |
| `--no-pdf` | PDF 를 만들지 않음. 기본은 `report.pdf` 생성 (`reportlab` 필요, 루트 `requirements.txt`에 포함) |

### 결과 (`outputs/graph/<실행시각>/`)

| 파일 | 내용 |
|---|---|
| `report.pdf` | **최종 보고서 PDF** (보고서 에이전트가 `final_markdown`으로 생성, 한글 폰트 자동 탐색) |
| `report.md` | 같은 보고서의 Markdown (`report_sections["final_markdown"]`) |
| `summary.md` | **실행 경로**(step 별 노드, 갱신한 키, 오류), 노드별 실행 방식과 상태, 공통 State 요약, PDF 경로 |
| `final_state.json` | 실행이 끝난 AppState 전체 |
| `trace.json` | 노드 실행 기록 (step, 노드, 갱신한 키, 오류) |
| `graph.mmd` | LangGraph 가 그린 부모 그래프 구조 (Mermaid) |

### 설계대로 돌았는지 확인하기

콘솔 첫 줄과 `summary.md`의 "실행 경로"를 보면 됩니다. **같은 step 번호의 노드는 병렬 실행**입니다.

```
실행 경로: START → technical → market + stakeholder + domain → synthesis → report → END
```

| step | 노드 | 갱신한 키 |
|---|---|---|
| 1 | technical | evidence_store, technical_findings |
| 2 | domain | domain_findings, evidence_store |
| 2 | market | evidence_store, market_findings |
| 2 | stakeholder | evidence_store, stakeholder_findings |
| 3 | synthesis | synthesis |
| 4 | report | quality_by_perspective, references, report_sections, retries, run_meta |

### 오프라인 실행 결과 (합성 fixture)

| 노드 | 상태 |
|---|---|
| technical | complete |
| market | complete |
| stakeholder | partial |
| domain | partial |
| synthesis | partial (매트릭스 12칸, 상충 9, 공유 근거 3, 보완 3, 요약 주장 14개) |
| report | needs_review |

보고서가 `needs_review`인 것은 정상입니다. fixture 에 일부러 넣은 잘못된 시장 주장("ITME는 처리량을 35.7% 높였다", CPU-offload 대비 최대값 표시 누락)이 보고서 본문까지 흘러갔고, 보고서 에이전트의 검사가 이를 잡았습니다. 에이전트가 연결된 상태에서도 품질 검사가 작동한다는 확인입니다.

---

## 4. 테스트

```bash
python -m unittest tests.graph.test_parent_graph -v
```

| 테스트 | 확인하는 것 |
|---|---|
| 실행 순서 | ① → ②③④ 같은 step(병렬) → ⑤ → ⑥ |
| 노드 오류 없음 | 모든 노드가 예외 없이 끝남 |
| 키 소유 | 각 노드가 자기 소유 키만 갱신 |
| 병렬 근거 병합 | ②③④가 동시에 쓴 `evidence_store`가 빠짐없이 합쳐짐 |
| 종합·보고서 생성 | `synthesis` 매트릭스와 `report_sections`가 만들어짐 |
| 에이전트 간 검사 연결 | fixture 의 조건 누락 수치를 보고서 검사가 잡음 |
| 실제 노드 연결 | 실제 시장·이해관계자 노드를 가짜 LLM·검색으로 꽂아도 그래프가 끝까지 돌고, 빈 결과도 종합·보고서가 한계로 처리 |
| 실제 도메인 노드 실패 | 실제 도메인 노드가 LLM 없이 실패해도 `failed` 결과로 끝나고 그래프는 종합·보고서까지 진행 |
| PDF 출력 | 보고서 노드가 `report.pdf`를 만들고 경로를 `run_meta["report"]["pdf_path"]`에 기록 |
| 재생 노드 | 그 관점이 인용한 근거만 돌려줌 |

---

## 5. 아직 없는 것

- **재실행 라우팅.** 평가 종합이 `retry_requests`를 남기지만, 부모 그래프에 그 관점을 다시 돌리는 조건부 엣지는 없습니다. 설계서 §8.1 은 재시도를 각 에이전트 내부 루프로 두므로 필수는 아닙니다.
- **① 기술 조사 실제 노드.** 에이전트 PR 전이라 fixture 재생 노드로 채웁니다.
- **체크포인트.** `build_graph(checkpointer=...)`는 받을 수 있지만 `main.py`에서는 쓰지 않습니다.

**PDF 폰트.** macOS 는 AppleGothic 을 자동으로 씁니다. 폰트를 못 찾으면 환경변수 `REPORT_PDF_FONT`에 TTF/TTC 경로를 지정하세요 (`agents/report/pdf.py`).
