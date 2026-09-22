# KV cache 다관점 평가 Multi-Agent

구조와 규칙은 [디렉터리 구조 설명.md](디렉터리%20구조%20설명.md)를, 남은 문제는 [ISSUE.md](ISSUE.md)를 먼저 읽어주세요.

## 구조 원칙

- 구현(웹 검색, 원문 수집, RAG, 품질 검사)은 에이전트마다 따로 둔다.
- 공유하는 것은 `graph/`의 State와 근거 형식 약속뿐이다.
- 각 에이전트는 `make_node()` 하나만 밖으로 내보내고, 그 안에서 내부 형식을 부모 State 형식으로 변환한다.

```
graph/                  공유 계약 + 연결
├── state.py            PipelineState (최초 설계)
├── team_state.py       EvaluationState (팀 설계서 v0.3), evidence_store reducer
└── build.py            add_node / add_edge 만

agents/
├── domain/             도메인 평가 (Tavily + BM25/dense RAG)
│   ├── __init__.py     make_node, DomainAgentDeps
│   ├── node.py         부모 State ↔ DomainLocalState 변환
│   ├── subgraph.py     plan_questions → retrieve → … → analyze
│   ├── state.py        도메인이 부모 State에 요구하는 확장 키
│   ├── prompts.py
│   ├── rag/  quality/  runtime/  evaluation/
└── stakeholder/        이해관계자 평가 (OpenAI Responses web_search)
    ├── __init__.py     make_node
    ├── node.py         부모 State ↔ StakeholderState 변환
    ├── subgraph.py     plan → search → extract → review 반복
    ├── models.py  backend.py  web.py  offline.py

scripts/                에이전트 단독 실행 (python -m scripts.run_stakeholder)
tests/agents/<이름>/     에이전트별 테스트
docs/                   에이전트별 설계 문서, State 설계
notebooks/  outputs/  data/
```

## 노드 연결

```python
from agents.domain import DomainAgentDeps, make_node as domain_node
from agents.stakeholder import make_node as stakeholder_node
from graph.build import build_graph

app = build_graph(
    ...,
    domain=domain_node(deps),
    stakeholder=stakeholder_node(legacy=True),  # graph/state.py 기준일 때
)
```

## 아직 정하지 않은 것

State 통일, 근거 형식, openai 버전 등 남은 문제는 [ISSUE.md](ISSUE.md)에서 관리합니다.

---

## 직접 테스트하기

모든 명령은 **레포 루트(`kongjinho/`)에서** 실행합니다.

### 0. 환경 설정 (처음 한 번)

Python 3.10 이상이 필요합니다.

```bash
cd kongjinho
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

용도에 따라 둘 중 하나를 설치합니다.

```bash
# (A) 오프라인 테스트만: 가볍고 빠름
pip install langgraph pydantic httpx openai beautifulsoup4 rank-bm25 pdfplumber pytest

# (B) 실제 API 실행, 노트북까지 전부: sentence-transformers(torch) 때문에 오래 걸림
pip install -r requirements.txt pytest
```

API 키가 필요한 실행(2단계)을 하려면 `.env`를 만듭니다. `.env`는 git에 올라가지 않습니다.

```bash
cp .env.example .env
# .env 를 열어 값 입력
# OPENAI_API_KEY=sk-...     두 에이전트 공통
# TAVILY_API_KEY=tvly-...   도메인 에이전트 웹 검색
```

### 1. API 키 없이 (비용 없음)

| 무엇을 | 명령 | 확인할 것 |
|---|---|---|
| 전체 테스트 | `python -m pytest tests` | `40 passed` |
| 도메인 테스트만 | `python tests/agents/domain/test_domain_agent.py` | `전체 통과` |
| 이해관계자 테스트만 | `python -m unittest tests.agents.stakeholder.test_stakeholder -v` | `OK` |
| 이해관계자 전체 흐름 (fixture 재생) | `python -m scripts.run_stakeholder --offline-fixture tests/agents/stakeholder/fixtures/web_replay.json --as-of 2026-09-21` | `상태: complete`, `outputs/stakeholder/<실행시각>/`에 결과 생성 |

- 도메인 테스트는 LLM과 네트워크 없이 근거 검증, 품질 검사, 입력 투영 규칙을 검사합니다.
- 이해관계자 fixture는 example.com을 쓰는 **합성 자료**입니다. 흐름이 끝까지 도는지만 확인할 수 있고, 실제 조사 결과는 아닙니다.

### 2. 실제 API로 (키 필요, 비용 발생)

#### 이해관계자 에이전트

```bash
# 가장 작은 규모: 검색 요청 1회
python -m scripts.run_stakeholder --ask-key --max-queries 1 --rounds 1 --revisions 0

# 기본 규모: 검색 요청 최대 8회
python -m scripts.run_stakeholder --ask-key
```

- `--ask-key`는 키를 화면에 표시하지 않고 입력받습니다. 환경변수 `OPENAI_API_KEY`가 있으면 생략해도 됩니다.
- 기본 모델은 `gpt-5.5`입니다. 계정에서 쓸 수 없으면 `--model <모델명>`으로 바꿉니다.
- 주요 옵션: `--as-of YYYY-MM-DD`(기준일), `--allowed-domain arxiv.org`(허용 도메인, 반복 가능),
  `--output-dir <폴더>`. 전체 목록은 `python -m scripts.run_stakeholder --help`로 볼 수 있습니다.
- 결과는 `outputs/stakeholder/<실행시각>/`에 저장됩니다.
  - `stakeholder_report.md`: 사람이 읽는 보고서
  - `team_state_update.json`: 부모 State에 합쳐질 업데이트
  - `research_trace.json`: 실제 검색 질의와 원문 기록

#### 도메인 에이전트

단독 실행 스크립트 대신 노트북으로 실행합니다. (B) 설치가 필요합니다.

```bash
jupyter notebook notebooks/06-domain-agent.ipynb
```

- 위에서부터 셀을 차례로 실행합니다. 첫 셀이 `.env`에서 키를 읽습니다.
- `OPENAI_API_KEY`는 필수입니다. 모델은 노트북의 `MODEL = "gpt-4o-mini"`에서 바꿀 수 있습니다.
- `TAVILY_API_KEY`가 없으면 검색 캐시(`data/search_cache/`)를 재생하는 오프라인 모드로 돕니다.
  캐시는 git에 올라가지 않으므로, 새로 받은 레포에서는 Tavily 키가 있어야 실제 결과가 나옵니다.
- 실행 결과는 노트북 셀 출력으로 확인합니다. status, 근거·주장·판정 개수, 품질 검사 결과, 인용 근거 순서로 나옵니다.

#### 도메인 검색 방식 비교 실험 (API 키 불필요, 인터넷 필요)

BM25 / dense / hybrid 검색 성능(Hit@k, MRR, 지연, 메모리)을 비교합니다. (B) 설치가 필요합니다.

```bash
python -m agents.domain.evaluation.ablation --embedding BAAI/bge-m3   # 세 방식 모두
python -m agents.domain.evaluation.ablation --embedding ""            # BM25만 (빠름)
```

- 처음 실행하면 논문 원문을 내려받아 `data/fetch_cache/`에 저장하고, 임베딩 모델(bge-m3)도 내려받습니다. 시간이 걸립니다.
- 결과는 `outputs/ablation.json`에 저장됩니다.

### 3. 아직 할 수 없는 것

- **전체 그래프 연결 실행**: 실행 진입점(`main.py`)이 없고 부모 State가 확정되지 않았습니다.
  State 설계서를 반영한 뒤 추가할 예정입니다. ([ISSUE.md](ISSUE.md) 1번, 4-2)
