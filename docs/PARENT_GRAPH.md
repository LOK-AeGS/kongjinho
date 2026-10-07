# 부모 그래프

설계서 v0.8 §8.1 아키텍처를 **Supervisor 패턴**으로 연결한 것입니다 (이전의 고정 순서 그래프를 대체).

```
START → supervisor ⇄ {technical, market, stakeholder, domain, synthesis, report, quality} → … → END
```

모든 노드의 유일한 출구는 supervisor 입니다(`graph/build.py`). supervisor 는 State(관점별 결과·근거 충분성·재작업 예산·품질 평가 결과)를
보고 다음 노드를 정합니다(`graph/supervisor.py`). 일반적인 경로는 technical → market·stakeholder·domain(`Send` 병렬) → synthesis → report → quality 이지만,
관점의 근거가 부족하거나 품질 평가가 미달이면 해당 관점·보고서만 다시 돌아 경로가 달라집니다. 판단 순서와 종료 보장은 README 의 "Supervisor 판단 순서" 참고.

---

## 1. 파일

| 파일 | 역할 |
|---|---|
| `graph/build.py` | Supervisor 중심 엣지 정의. 노드 함수는 밖에서 받아 `as_worker` 로 감쌈 |
| `graph/supervisor.py` | Supervisor 노드(허용 행동 계산 → LLM 제안 → guard/규칙 폴백), 라우터(`Send` 병렬), 결정 로그 |
| `graph/sufficiency.py` | 근거 충분성 기준과 재작업 예산 |
| `graph/workers.py` | 워커 래퍼: 재작업 힌트 주입·예외 격리·시도 횟수 기록 |
| `graph/quality.py` | 보고서 품질 평가 노드 (Groundedness·중립성·편향·관점 커버리지, 선택적 LLM Judge) |
| `graph/state.py` | 공통 State `AppState`(페이로드 + 제어 메타데이터), reducer, `create_initial_state` |
| `graph/stubs.py` | **임시 노드**: 아직 AppState 형식 노드가 없는 자리를 합성 fixture 재생으로 채움 |
| `main.py` | 실행 진입점. 노드 조립 → `build_graph` → 실행 → 결과 저장 |
| `tests/graph/test_parent_graph.py` | 부모 그래프 통합 테스트 (API·네트워크 없음) |
| `tests/graph/test_supervisor.py` | 라우팅·재작업·종료·품질 Loop·LLM guard·State 계약(trace_id·sqlite 재개·reducer)·오프라인 `main.py` 종단 테스트 |

규칙대로 엣지는 `graph/build.py`에만 있고, `main.py`는 각 에이전트의 `make_node()`만 가져다 씁니다.

---

## 2. 노드별 연결 상태 (2026-09-22, 여섯 에이전트 모두 연결)

| 노드 | 기본 실행 (`python main.py`) | `--live` 실행 |
|---|---|---|
| ① technical | fixture 재생 | `agents.technical`: Pool A 고정 PDF RAG(BM25 + BGE-M3 + RRF) + Tavily, gpt-4.1-nano, 코드로 TRL Gate 판정 |
| ② market | fixture 재생 | `agents.market`: gpt-4.1-mini/gpt-4.1 + Tavily |
| ③ stakeholder | fixture 재생 | `agents.stakeholder`: OpenAI 웹 검색 + 원문 fetch·인용 검증 (`STAKEHOLDER_MODEL`) |
| ④ domain | fixture 재생 | `agents.domain`: gpt-4o + Tavily, 긴 문서는 bge-m3 임베딩 (`DOMAIN_MODEL`, `DOMAIN_EMBEDDING`) |
| ⑤ synthesis | 실제 노드, 템플릿 서술 | `agents.synthesis` + gpt-4.1 서술 |
| ⑥ report | 실제 노드, deterministic | `agents.report`: gpt-4o-mini |

**공통 입력.** 기술 조사 에이전트는 `selected_tech`가 자기 고정값과 정확히 같아야 실행되므로(`agents/technical/config.py`),
`main.py`는 그 팀 고정 입력(`DEFAULT_REQUEST`, `DEFAULT_SELECTED_TECH`)을 부모 그래프 전체의 입력으로 씁니다.
`corpus_manifest`는 Pool A 고정 코퍼스 목록(`data/technical/manifest.json`)으로 채웁니다.

**fixture 재생.** `tests/agents/synthesis/fixtures/appstate_sample.json`(합성 자료)에서 그 관점의 결과와 인용 근거만 돌려줍니다.
D1·D3 인용 외의 URL·수치는 가짜라서, 재생 노드가 하나라도 있으면 결과에 "실제 조사가 아님" 경고가 붙습니다.
API 비용 없이 연결을 확인하는 용도입니다.

---

## 3. 실행

```bash
# 전부 오프라인 (API 키 불필요, 비용 없음)
python main.py

# 일부만 실제 실행 (.env 의 OPENAI_API_KEY 사용, 비용 발생)
python main.py --live synthesis
python main.py --live synthesis,report

# 여섯 노드 전부 실제 실행 (technical·market·domain 은 TAVILY_API_KEY 도 필요)
python main.py --live all

# 진행 상황과 중간 결과를 자세히 보기
python main.py --live all --debug
```

| 옵션 | 내용 |
|---|---|
| `--live` | 실제로 실행할 노드 (쉼표 구분): `all` 또는 `technical`, `market`, `stakeholder`, `domain`, `synthesis`, `report` |
| `--debug` | 노드가 끝날 때마다 주장·공백·요약 문장·위반 샘플까지 콘솔에 출력 |
| `--rounds` | 시장·도메인의 최대 검색 라운드 1 또는 2 (기본 1). 기술 조사는 자체 고정값(2)을 씀 |
| `--as-of` | 조사 기준일 (기본: 팀 고정 입력의 2026-09-22) |
| `--fixture` | 재생 노드가 쓸 AppState JSON |
| `--output-dir` | 결과 폴더 (기본 `outputs/graph`) |
| `--no-pdf` | PDF 를 만들지 않음. 기본은 `report.pdf` 생성 (`reportlab` 필요, 루트 `requirements.txt`에 포함) |

### 결과 (`outputs/graph/<실행시각>/`)

| 파일 | 내용 |
|---|---|
| `report.pdf` | **최종 보고서 PDF** (보고서 에이전트가 `final_markdown`으로 생성, 한글 폰트 자동 탐색) |
| `report.md` | 같은 보고서의 Markdown (`report_sections["final_markdown"]`) |
| `summary.md` | **실행 경로**(step 별 노드, **소요 시간**, 갱신한 키, 오류), 노드별 실행 방식과 상태, 공통 State 요약, PDF 경로 |
| `steps/<단계>_<노드>.json` | **노드별 중간 결과**: 그 노드가 반환한 값 전체와 실행 기록. 예: `02_market.json` |
| `final_state.json` | 실행이 끝난 AppState 전체 |
| `trace.json` | 노드 실행 기록 (step, 노드, 소요 시간, 갱신한 키, 오류) |
| `graph.mmd` | LangGraph 가 그린 부모 그래프 구조 (Mermaid) |

### 실행 중 진행 표시와 중간 결과

노드가 끝나는 즉시 콘솔에 한 줄씩 나옵니다. 같은 `[단계]` 번호는 병렬 실행입니다.

```
  [1] ✓ technical       0.0초  status=complete, 판정 2칸, 주장 2개, 공백 0개, 근거 3건
  [2] ✓ domain          0.0초  status=partial, 판정 4칸, 주장 1개, 공백 1개, 근거 5건
  [2] ✓ market          0.0초  status=complete, 판정 4칸, 주장 3개, 공백 0개, 근거 6건
  [2] ✓ stakeholder     0.0초  status=partial, 판정 2칸, 주장 0개, 공백 1개, 근거 2건
  [3] ✓ synthesis       0.0초  status=partial, 매트릭스 12칸, 상충 9, 보완 3, 요약 주장 14개, 검사 passed
  [4] ✓ report          0.1초  status=needs_review, 섹션 16개, 참고문헌 10건, 위반 1건
```

- `--debug`를 주면 각 줄 아래에 주장·공백·한계, 평가 종합의 내부 경로와 요약 문장, 보고서 위반 사항 샘플이 붙습니다.
- 노드가 반환한 값 전체는 `steps/<단계>_<노드>.json`에 바로 저장됩니다. 실행이 중간에 멈춰도 그때까지 끝난 노드의 결과는 남습니다.
- 오류가 난 노드는 `✗`와 함께 오류 내용이 출력됩니다.

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
| 실제 기술 조사 노드 | 가짜 retriever·Tavily·analyzer 로 꽂아도 1단계에서 끝까지 돌고 TRL 칸을 만듦 |
| 공통 입력 | 초기 State 가 팀 고정 입력과 Pool A manifest 를 씀 |
| 중간 결과 콜백 | 노드마다 한 번씩 반환값과 함께 호출, 소요 시간 기록 |
| PDF 출력 | 보고서 노드가 `report.pdf`를 만들고 경로를 `run_meta["report"]["pdf_path"]`에 기록 |
| 재생 노드 | 그 관점이 인용한 근거만 돌려줌 |

---

## 5. 아직 없는 것

- **Supervisor 전환 후 실제 API 실행 기록.** 오프라인 테스트로 라우팅·재작업·종료는 확인했지만, `gpt-5.5`로 `--live all` 을 돌린 결과는 `docs/LIVE_RUN.md`에 있고, LangSmith 캡처(`tracing-*.png`)는 계정에서 직접 찍어야 합니다. 모델 접근이 안 되면 규칙 라우팅으로 대체되어 끝까지 돕니다(`source=fallback`).
- (해결됨) 모든 워커가 `rework_hint` 를 읽습니다: 기술 조사·시장·도메인은 보완 질의/계획 힌트로, 이해관계자는 gap 쌍만 재검색 후 결과 병합, 보고서는 미달 사유를 프롬프트로 받고 수정 횟수를 초기화합니다.
- **체크포인트 파일 저장.** `langgraph-checkpoint-sqlite` 가 없으면 메모리에만 저장됩니다.

**PDF 폰트.** macOS 는 AppleGothic 을 자동으로 씁니다. 폰트를 못 찾으면 환경변수 `REPORT_PDF_FONT`에 TTF/TTC 경로를 지정하세요 (`agents/report/pdf.py`).
