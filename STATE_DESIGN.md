# KV cache Multi-Agent State 설계

> 이 문서는 최초 PipelineState 설계 기록입니다. 팀 설계서 v0.3 기준의 현재 연결 계약은
> `team_state.py`와 `README.md`의 “팀 그래프 연결”을 따릅니다.
> 이해관계자 노드는 selected_tech/domain을 입력받아 stakeholder_eval/evidence_store/errors를 반환합니다.
> evidence_store는 dict + idempotent merge이며, not_found와 검색 로그는 stakeholder_eval.search_outcomes에 보존합니다.
> 아래 stakeholder_findings 방식은 legacy_stakeholder_agent에만 해당합니다.

앞서 작성한 전체 아키텍처와 6개 에이전트 다이어그램에 대응한다.
기술 선정은 사람이 수행하며, 공유 문서 인덱스 준비 이후 initial_state()를 호출한다.

## 부모 그래프 State

| 키 | 내용 | 작성자 | 주요 독자 |
|---|---|---|---|
| run_id | 실행 식별자 | 초기화 | 전체 |
| request | SW/HW 기술·선정 사유·도메인·기준일·반복 한도 | 초기화 | 전체 |
| corpus_manifest | 문서·선정 페이지·임베딩 선정 사유·인덱스 위치 | 초기화 | 기술·시장·도메인 |
| technical_findings | 원리·성능 조건·한계·TRL·근거 | 기술 조사 | 세 평가·종합·보고서 |
| market_findings | 시장 관련 주장·채택 단계·근거 | 시장 평가 | 종합·보고서 |
| stakeholder_findings | 관계자 입장·소속·편향·근거 | 이해관계자 평가 | 종합·보고서 |
| domain_findings | 도메인 요구사항·적합 조건·제약·근거 | 도메인 평가 | 종합·보고서 |
| synthesis | 비교 매트릭스·일치/상충/보완 관계·시사점 | 평가 종합 | 보고서 |
| report | SUMMARY·본문·실제 인용·파일 경로·검토 상태 | 보고서 생성 | 사용자 |

## 업데이트 규칙

시장 노드는 `return {"market_findings": result}` 형태로 반환한다.
입력 state를 직접 수정하거나 `return {**state, "market_findings": result}`로
전체 상태를 반환하지 않는다. 병렬 분기가 읽기 전용 필드까지 동시에 반환하면
같은 키에 대한 동시 업데이트가 발생한다.

현 구조는 각 최상위 결과 키에 작성자가 하나이므로 커스텀 reducer가 필요 없다.
모든 근거를 하나의 전역 리스트에 병렬 추가하지 않고 각 결과 안에 보관한다.
종합/보고서는 네 결과의 evidence 목록을 읽어서 근거를 모은다.
향후 하나의 에이전트 내에서 SW·HW까지 부모 그래프에 병렬 쓰도록 변경하면
키를 추가로 분리하거나 해당 채널에 reducer를 설계해야 한다.

LangGraph 공식 참고: https://docs.langchain.com/oss/python/langgraph/graph-api

## 에이전트 내부 State

AgentLocalState는 검색 질문, 검색 결과, 초안, 검색 횟수, 수정 횟수,
근거 충분성, 품질 검사 결과와 다음 행동을 보관한다.
부모 State와 별도의 서브그래프로 사용하고, wrapper 함수에서 필요한 입력을 구성한 뒤
최종 결과만 부모의 전용 필드로 반환한다. 서브그래프를 단순 등록만 해서
입출력이 자동 변환되는 구조는 아니다.

- 최초 검색을 포함해 search_rounds_used < max_search_rounds일 때만 추가 검색한다.
- 최초 작성 이후 revision_rounds_used < max_revision_rounds일 때만 수정한다.
- 부족한 근거를 끝내 확보하지 못하면 completion.status=partial과 gaps를 반환한다.
- 실패를 복구해 계속 진행한다면 completion.status=failed와 errors를 반환한다.
- 잡지 않은 예외는 그래프 실행을 중단한다. State 자체는 예외를 처리하지 않는다.
- None은 미완료 상태이고, partial/failed 결과 객체는 제한사항이 있는 종료 상태다.
- 종합은 결과가 None인지 확인하고 partial/failed의 한계를 최종 보고서에 반영한다.
- 보고서는 검사 실패 상태로 수정 한도에 도달하면 quality_status=needs_review로 반환한다.

## 근거 추적 계약

Claim은 주장, 대상 기술, 실험/적용 조건, evidence_ids와 불확실성을 가진다.
Evidence는 URL, 제목, 작성자, 날짜, 페이지/절, 짧은 원문을 가진다.
ID는 technical:claim:001, market:ev:001처럼 에이전트 접두사를 붙여 실행 내에서 고유하게 만든다.
재시도해도 같은 근거의 ID는 유지하는 것이 좋다.

각 에이전트의 주장은 자체 근거 또는 앞 단계에서 받은 기술 조사 근거를 참조할 수 있다.
검증할 때는 자체 결과와 실제로 전달받은 상위 결과를 합친 ID 집합을 사용한다.
시장·이해관계자·도메인 사이에는 서로의 동시 실행 결과를 참조하지 않는다.
종합의 relations.claim_ids는 관점 간 논쟁에 참여하는 원래 주장들을 연결한다.
종합의 implications는 기존 근거 ID를 참조하고, 새 사실을 임의로 추가하지 않는다.

노드의 결과 검증 단계에서 다음을 검사해야 한다. TypedDict가 자동 수행하지 않는다.

1. direct_evidence 주장은 하나 이상의 유효한 근거를 참조한다.
2. 추론은 conditions와 uncertainty로 전제·한계를 밝힌다.
3. TRL은 1~9 또는 None이며, 추정 사유와 한계를 포함한다.
4. cited_evidence_ids는 실제 본문에 사용한 근거와 일치한다.
5. 참고문헌은 cited_evidence_ids로 필터링하고 같은 출처를 중복 기재하지 않는다.

## 저장 경계와 검증 범위

State에는 JSON으로 저장 가능한 값만 넣는다. LLM 클라이언트, 검색 도구,
벡터 DB 연결 객체, PDF 전체 바이트와 임베딩 벡터는 State 밖의 런타임 의존성으로 관리한다.
corpus_manifest에는 인덱스 위치와 문서 메타데이터만 저장한다.

initial_state()는 기술명/선정 사유, 날짜, 도메인, 반복 한도, 문서 ID 중복,
선정 페이지 범위와 총 200페이지 제한을 검사한다. 파일 존재, 모델의 공개 라이선스,
출처의 진위나 모든 중첩 필드의 자료형을 검증하는 함수는 아니다.

state.py는 Python 표준 라이브러리만 사용한다. graph_wiring.py는 LangGraph가 필요하며,
실제 6개 에이전트 함수를 주입해야 한다. 검색·LLM·보고서 생성 구현은 포함하지 않는다.
