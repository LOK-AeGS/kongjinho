# KV cache 다관점 평가 Multi-Agent

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

- 부모 State: `graph/state.py`(PipelineState)와 `graph/team_state.py`(EvaluationState) 중 하나로 통일
- 근거 형식: 도메인은 `evidence_id`/`locator`, 이해관계자는 `id`/`page_or_locator`.
  둘 다 `evidence_store`에 쓰려면 형식과 reducer를 하나로 맞춰야 한다.
- lock 파일: `requirements.lock`(도메인), `requirements.lock.txt`(이해관계자)가 따로 있다.
  도메인 lock은 `openai==3.x`, 이해관계자는 `openai<3` 기준으로 작성·테스트했다.

## 테스트

```bash
python -m pytest tests          # 전체
python tests/agents/domain/test_domain_agent.py
python -m unittest tests.agents.stakeholder.test_stakeholder
```
