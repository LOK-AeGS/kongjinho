# 이해관계자 평가 에이전트 (간소화 버전)

`agents/stakeholder_eval.py` 하나짜리 파일입니다. `agents/stakeholder/`(이해관계자 **검색** 에이전트,
`docs/STAKEHOLDER_AGENT.md`)와는 완전히 별개의 새 에이전트로, 같은 목적(이해관계자 반응 조사)을
훨씬 단순한 구조로 구현했습니다. 프로그래밍에 익숙하지 않아도 코드를 위에서 아래로 읽으면 전체 흐름을
따라갈 수 있도록 만드는 것을 최우선 목표로 삼았습니다.

## 무엇을 하는 에이전트인가

**입력:** 우리 팀이 조사 중인 기술 2개(sw/hw), 도메인(예: "datacenter"), 그리고 기술 조사 에이전트가
먼저 만들어 둔 결과(`technical_findings`, 선택 사항).

**하는 일:** 기술마다 4가지 이해관계자 그룹(경쟁 기술 진영 / 운영자·서빙 엔지니어 / 공급사 / 투자자)에게
"이 기술에 대해 뭐라고 말했는지" 실제로 웹에서 검색해서 찾아냅니다. 찾은 발언을 지지/반대/중립으로
분류하고, 발언자가 그 기술과 이해관계가 있어 보이면 표시해 둡니다.

**출력:** 팀 공통 계약 파일인 [graph/state.py](../graph/state.py)의 `AppState.stakeholder_findings`
형식으로 결과를 돌려줍니다.

## 5단계 흐름

원래 흐름도(`docs/STAKEHOLDER_AGENT.md`에 있는 다이어그램)의 상자 이름을 그대로 따라가되, 실제 코드에서는
5개의 함수 호출로 정리했습니다. `agents/stakeholder_eval.py`를 위에서부터 읽으면 이 순서 그대로 나옵니다.

| 순서 | 원래 흐름도 상자 | 이 버전의 함수 | 하는 일 |
|---|---|---|---|
| 1 | 이해관계자별 검색 계획 | `build_search_plan` | 기술 2개 × 그룹 4개 = 8개의 검색어를 만듭니다. |
| 2 | 웹 검색·원문 열람 / 발언 맥락 추출 / 분류 | `search_group_opinions` | OpenAI에게 "직접 웹에서 찾아보고, 찾은 내용만 정해진 형식으로 정리해줘"라고 한 번에 요청합니다. |
| 3 | 편향 점검 | `flag_bias` | 발언자 이름·소속에 그 기술 이름이 들어 있으면 "자기 홍보일 수 있음" 메모를 붙입니다. |
| 4 | 관점이 누락되었는가? | `find_missing_groups` (+ `run_stakeholder_eval` 안의 재검색) | 8개 조합 중 하나도 못 찾은 게 있으면, 검색어를 살짝 바꿔 **딱 한 번만** 더 찾아봅니다. |
| 5 | 출력: stakeholder_findings | `to_perspective_findings` | 찾은 발언들을 팀 공통 형식(claims/records/gaps)으로 정리합니다. |

전체를 순서대로 실행하는 함수가 `run_stakeholder_eval`이고, 이걸 팀 그래프에 연결하는 어댑터가
`make_node`입니다. `make_node()`가 반환하는 함수를 `graph/build.py`의 `stakeholder=` 자리에 넣으면
됩니다 (기존 `agents/stakeholder`의 노드를 대체합니다).

## 원래 설계와 달라진 점

기존 `agents/stakeholder/` 에이전트는 "찾은 내용이 실제로 맞는지"를 코드로 다시 한번 검증하는 정교한
안전장치를 많이 갖고 있습니다. 이 버전은 그 안전장치들을 의도적으로 빼서 코드량을 크게 줄였습니다.
무엇을 얻고 무엇을 잃는지 정리하면 다음과 같습니다.

| 원래 설계 (`agents/stakeholder/`) | 이 버전 | 왜 이렇게 단순화했나 | 잃는 것 |
|---|---|---|---|
| 검색으로 찾은 URL을 직접 다시 방문(fetch)해서 robots.txt 확인, 접근 차단(SSRF) 방지, 페이지를 파일로 저장(snapshot 캐시) | 별도 재방문 없이 OpenAI 웹검색 도구의 결과를 그대로 신뢰 | 원문을 직접 내려받고 검사하는 코드(`web.py`)가 파일 전체 분량이라, 이걸 빼는 것만으로 복잡도가 크게 줄어듦 | 검색 도구가 실수로 존재하지 않는 페이지 내용을 요약했을 가능성을 코드로는 걸러내지 못함 |
| 인용문(quote)이 실제 원문의 특정 block에 글자 그대로 있는지 해시값·위치(locator)로 대조 검증 | LLM이 "이게 원문 인용문이다"라고 보고한 내용을 그대로 사용 | 원문 재방문을 빼기로 한 이상 대조 검증도 같이 뺄 수밖에 없음 | 인용문이 살짝 다듬어지거나(paraphrase) 부정확할 위험이 있음 — 중요한 결정에는 원문 링크를 직접 확인해야 함 |
| LangGraph 서브그래프: `plan → search → extract → review` 후 조건부 엣지로 `research`(재검색)/`revise`(재추출)를 한도까지 반복 | 그래프 없이 위→아래로 실행되는 일반 함수. 누락된 조합만 **최대 1회** 재검색 | LangGraph의 조건부 엣지·상태 누적 방식은 그래프 개념을 모르면 실행 순서를 눈으로 따라가기 어려움. 일반 함수는 코드를 그냥 순서대로 읽으면 됨 | 검색 예산이 넉넉해도 2회 이상 반복 탐색하지 않아, 원래 버전보다 누락(Gap)이 더 많이 남을 수 있음 |
| 편향 점검을 LLM에게 별도로 맡김(공급자 홍보성 등 판단) | 발언자/소속 문자열에 기술 이름이 포함되는지만 검사하는 규칙 | 규칙 하나로 대부분의 명백한 자기 홍보는 걸러지고, LLM 호출을 추가로 쓰지 않아도 됨 | "은근히 우호적인" 미묘한 이해관계는 못 잡아냄 (예: 투자자가 소속을 밝히지 않은 경우) |
| 발언을 못 찾은 이유를 `not_found`(진짜 없음) / `blocked`(검색·검증 실패) / `unsearched`(예산 부족으로 아직 안 함) 3가지로 구분 | 모두 `Gap`(찾지 못함) 하나로 통합 | 원인을 세분화하려면 각 단계의 실패를 추적하는 코드가 더 필요한데, 이 버전은 그 단계 자체(원문 재검증 등)가 없어서 구분할 정보가 없음 | "정말 의견이 없는 것"과 "검색이 막힌 것"을 구분하지 못함 — Gap이 있으면 사람이 원인을 직접 확인해야 함 |
| 실행마다 원문 HTML·검색 로그를 `outputs/stakeholder/<실행시각>/`에 저장 | 저장하지 않음 (검색 로그만 결과 안에 남김) | 원문을 직접 받아오지 않으므로 저장할 파일 자체가 없음 | 나중에 "그때 그 페이지에 정말 그렇게 쓰여 있었는지" 재현해서 확인할 수 없음 |

한 줄 요약: **속도와 코드 가독성을 위해 "직접 원문을 검증하는 안전장치"를 포기하고 LLM의 웹 검색
결과를 신뢰하는 쪽을 택했습니다.** 결과를 실제 의사결정에 쓸 때는 `stakeholder_findings.records[].evidence_ids`
로 연결되는 `evidence_store`의 `url`을 열어 직접 한 번 확인하는 것을 권장합니다.

## 검증 기록

실제 `OPENAI_API_KEY`로 1개 조합(`sw` × `데이터센터 운영자 / 서빙 엔지니어`)을 호출해 확인했습니다.
처음에는 `max_output_tokens=4000`으로 두었더니 JSON 응답이 중간에 잘려 파싱 오류가 났습니다
(`agents/stakeholder/`의 검색 에이전트도 같은 이유로 3000→8000으로 올린 이력이 있습니다,
`docs/STAKEHOLDER_AGENT.md` 참고). 이 버전도 `8000`으로 올리고, 지시문에 "opinions는 최대 3개까지만"을
추가해 응답 길이 자체를 줄이는 방식으로 해결했습니다. 이후 실제 웹 검색 결과 3건을 정상적으로 찾아
`PerspectiveFindings` 형식으로 변환하는 것까지 확인했습니다.

## 실행 방법

실제 OpenAI API를 호출하므로 비용이 발생합니다(기술 2개 × 그룹 4개 = 최소 8회, 누락이 있으면 최대
16회의 `responses.parse` 호출). `OPENAI_API_KEY` 환경변수가 필요합니다.

```python
from agents.stakeholder_eval import make_node

node = make_node()  # model="gpt-4.1-mini" 기본값, client 생략 시 실제 OpenAI() 클라이언트 생성
result = node({
    "selected_tech": {"sw": {"name": "..."}, "hw": {"name": "..."}},
    "domain": "datacenter",
    "technical_findings": None,
})
print(result["stakeholder_findings"])
```

네트워크 없이 로직만 확인하려면 `tests/agents/test_stakeholder_eval.py`를 참고하세요. `client` 자리에
고정된 응답을 돌려주는 stub을 넣어 실제 API 호출 없이 전체 흐름을 테스트합니다.

```bash
python -m unittest tests.agents.test_stakeholder_eval -v
```
