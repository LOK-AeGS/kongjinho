# 이해관계자 평가 에이전트 v0.3

데이터센터 LLM 추론에서 DeepSeek-V2 MLA와 ITME CXL-Hybrid에 대한 공개 반응을 조사합니다.
설계 v0.3의 Live Web Retrieval·원문 검증·근거 병합 계약을 반영했습니다.
Word 설계 문서와 다른 팀 에이전트 구현은 수정하지 않았습니다.

## 변경 내용

- HTML 원문 수집·실행별 snapshot/cache, 접근 상태와 검색 로그
- 원문 block locator·quote·content hash·날짜·수치/단위의 결정적 검사
- `dict[evidence_id, Evidence]`와 idempotent merge reducer
- 반증 부재를 허용하는 `not_found`, 접근/검증 실패인 `blocked`, 미검색 `unsearched` 구분
- 하나의 원문 근거를 여러 claim·관점에서 재사용할 때 연결 보존
- v0.3 State 키 및 API 없는 offline fixture 실행

## 실행

Python 3.10 이상. 기존 프로젝트 가상환경을 사용할 수 있습니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_stakeholder.py --ask-key --max-queries 8
```

현재 모듈의 기본 선택은 GPT-5.5 + OpenAI Responses `web_search`이며, 팀 전체 공급자 확정과는 별개입니다.
`--model` 또는 `OPENAI_MODEL`로 모델을 지정합니다. 지원/접근 권한은 API 계정에 따라 다릅니다.
`--ask-key`는 키를 화면에 표시하거나 파일에 저장하지 않습니다.
이미 OPENAI_API_KEY가 설정되어 있으면 생략할 수 있습니다.

**비용 한도:** 기본값은 전체 실행에서 검색 API 요청 최대 8회, 요청당 검색 도구 최대 2회입니다.
원문은 질의당 최대 5개 URL을 확인합니다. 구조화는 검색 라운드당 1회, 검증 수정은 최대 1회 추가합니다.
SDK의 일시 오류 재시도 1회는 이 논리 요청 횟수 외에 발생할 수 있습니다.
이는 달러 단위 지출 상한이 아닙니다. 계정의 별도 예산 설정과 함께 사용하세요.
`--max-queries 1 --rounds 1 --revisions 0`으로 연결 시험 규모를 줄일 수 있습니다.
6개 기술/그룹 조합(2기술 × 3그룹: 경쟁 기술 진영/도입 기업·개발자/투자·산업 관계자)보다
검색 예산이 적으면 일부 조합은 unsearched가 됩니다.

```bash
python run_stakeholder.py --ask-key --max-queries 1 --rounds 1 --revisions 0
python run_stakeholder.py --as-of 2026-09-21 --allowed-domain github.com --allowed-domain arxiv.org
```

기간 정책은 시장·채택 자료 최근 24개월 우선, 표준·원전은 기간 제한 없음입니다.
검색 기간은 질의 지시로 전달하며 공급자 서버의 엄격한 날짜 필터는 아닙니다.
최종 기준일은 원문에서 확인한 발행일로 검사합니다. 날짜 미상은 partial로 남깁니다.
허용 도메인 기본값은 빈 목록(공개 웹 전체)이며, 지정하면 검색과 fetch 양쪽에 적용합니다.
검색 요청과 원문 HTTP 요청 간격은 각각 최소 1초입니다. 원문 timeout은 20초,
API timeout은 120초입니다. 원문 실패는 무제한 재시도하지 않습니다.

## API 없이 실행

```bash
python run_stakeholder.py --offline-fixture tests/fixtures/web_replay.json --as-of 2026-09-21
python -m unittest discover -s tests -v
```

제공된 fixture는 example.com을 사용하는 **합성 테스트 자료**입니다.
API와 네트워크를 호출하지 않으며 보고서 첫머리에 실제 조사 결과가 아님을 표시합니다.

## 결과

`outputs/stakeholder/<실행시각>/`에 저장합니다.

| 파일 | 내용 |
|---|---|
| team_state_update.json | stakeholder_eval / evidence_store / errors 업데이트 |
| stakeholder_findings.json | 단독 실행 상세 결과·검증된 근거·검색 결과 상태 |
| stakeholder_report.md | 클릭 가능한 출처와 한계·not_found 목록 |
| research_trace.json | 질의·공급자·실제 검색 action·원문 block·실행 설정 |
| evidence_cache/ | 실제 실행의 원문 HTML, 메타데이터 JSON, 검색 기록 |

캐시는 실행별 저장 및 같은 실행 내 URL 재사용 방식입니다. 실행 간 자동 재사용은 하지 않습니다.
원문 캐시는 고정 코퍼스 인덱스에 자동 편입하지 않습니다. 웹 자료의 200페이지 예산 포함 여부는 팀 정책 미확정입니다.
`temperature`와 `seed`는 미지정(API 기본값)으로 기록하며, 고정됐다고 주장하지 않습니다.

## 팀 연결 계약

```python
from langgraph.graph import StateGraph
from team_state import EvaluationState
from stakeholder.agent import default_request, stakeholder_agent

builder = StateGraph(EvaluationState)
builder.add_node('stakeholder', stakeholder_agent)
config = default_request()
state = {
    'selected_tech': {'sw': config['sw'], 'hw': config['hw']},
    'domain': 'datacenter',
    'as_of_date': config['as_of_date'],
    'tech_profiles': {},
    'evidence_store': {},
}
# 팀 그래프의 edges/compile은 팀 담당자가 연결합니다.
update = stakeholder_agent(state)
```

공통 기술 프로필만 읽고, 다른 관점의 *_eval·중간 결론·Judge 판단은 프롬프트로 전달하지 않습니다.
출력은 stakeholder_eval, evidence_store, errors 3개 키입니다.
`quality_by_perspective`, `failed_perspectives`, `retries`, dict형 `references`는 v0.3 State에 정의했습니다.
이 노드는 공통 Judge의 점수·실패 목록·재시도 횟수를 덮어쓰지 않습니다.
TRL의 합류 후 실행, 공통 gate/Judge, 실패 관점만 최대 2회 재실행하는 최상위 라우팅은 팀 그래프 담당 범위입니다.
기존 graph_wiring.py는 이전 구조용이며 새 그래프가 구현된 것으로 간주하면 안 됩니다.

## 근거와 미발견 기록

근거 ID는 최종 URL + 원문 locator + 공백 정규화 quote의 SHA-256입니다.
content_hash는 추출된 원문 block 전체의 SHA-256이고, snapshot JSON에 같은 값을 보존합니다.
HTML은 원본 바이트로 저장합니다. locator는 이 snapshot 안의 block:0001 형식입니다.
같은 ID 재실행은 최신 accessed_at의 검증 결과를 사용합니다. 동일 timestamp는 안정적인 정렬로 결정합니다.
claim_ids, perspectives, bindings를 병합해 여러 주장·관점의 연결과 각 입장을 잃지 않습니다.
단일 claim_id/perspective/stance는 최신 대표값이고, 전체 연결은 bindings를 사용합니다.

근거 필드: id, claim_id, doc_id, title, author_or_org, source_type,
primary_or_secondary, direct_or_proxy, url, published_at, accessed_at,
page_or_locator, quote, stance, perspective, content_hash 및 위 연결 확장 필드.

`not_found`는 가짜 quote를 가진 Evidence를 생성하는 대신 stakeholder_eval.search_outcomes에 저장합니다.
기술·그룹·탐색한 입장·query_ids·검색 범위를 함께 기록합니다.
- found: 검증된 원문에서 해당 기술 직접 반응을 확보
- not_found: 지정 검색이 완료되고 확인 가능한 결과를 처리했지만 해당 반응을 확보하지 못함
- blocked: 검색·원문 접근·구조화·검증 미완료
- unsearched: 질의 예산 부족 등으로 아직 조사하지 않음

긍정/부정 16개 슬롯 강제는 제거했습니다. complete는 정해진 범위의 수집·검사가 끝났다는 뜻이며,
검색 부재가 실제 의견의 부재를 증명하거나 보고서 품질 통과를 뜻하지는 않습니다.
confidence는 직접 근거를 확보한 기술/그룹 비율이며, 진실성 확률이나 Judge 점수가 아닙니다.

## 제약과 검증 결과

현재 fetcher는 공개 HTML을 지원합니다. PDF·JS 렌더링 필수 페이지는 지원하지 않으며
실패 상태로 기록합니다. 확인 가능한 paywall을 기록하고 우회하지 않습니다.
숨겨진 구독 장벽을 모두 탐지할 수는 없습니다. 링크의 원문 유형과 수집 품질은 제출 전 점검하세요.

quote의 원문 포함 여부, locator, hash와 숫자/단위의 문자열 존재를 검사합니다.
수치의 인과관계·비교 기준·번역 정확성·실제 발언자·primary/proxy 분류까지 코드가 증명하지는 않습니다.
이 부분은 qualitative Judge와 사람 검토가 필요합니다. 검색 요약문은 구조화 입력에서 제외합니다.

15개 오프라인 테스트 통과: 원문 검증, 반증 미발견, 접근 실패, 예산, 날짜·수치,
근거 다중 연결, reducer의 멱등성/결합성, 병렬 병합, 프롬프트 분리, 캐시 처리.
이전 실API 시험에서 환경 키의 AuthenticationError가 확인됐으며 이번 수정에서는 재호출하지 않았습니다.
실제 검색 end-to-end 성공은 유효한 API 키로 확인해야 합니다.

[OpenAI 웹 검색](https://developers.openai.com/api/docs/guides/tools-web-search)
