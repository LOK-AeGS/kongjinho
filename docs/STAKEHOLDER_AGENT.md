# 이해관계자 평가 에이전트 (`agents/stakeholder`)

## 하는 일
선정 기술 2개(SW·HW) × 관계자 그룹 3개(경쟁 기술 진영 / 도입 기업·개발자 / 투자·산업 관계자)의 데이터센터 관점 공개 반응을
조사해 지지(support)·반대(counter)·중립(neutral)으로 기록한다. RAG 없이 OpenAI `web_search`로 URL 을 찾고, **원문 HTML 을 직접 가져와** 검증한다.

## 왜 원문 검증형인가 (환각 문제)
`agents/stakeholder_eval.py`(간소화 버전)는 LLM 이 검색 결과를 읽고 발언자·인용문·URL·날짜를 한 번에 채우고 코드 검증이 없었다.
그래서 존재하지 않는 인용문이나 URL 이 `evidence_store`에 그대로 들어갈 수 있었다(`page_or_locator="web_search_result"`, 가짜 content_hash).
`agents/stakeholder`는 아래를 **코드가** 확인하고, 통과한 항목만 근거로 인정한다.

| 검사 | 통과 조건 |
|---|---|
| 원문 접근 | fetch 상태가 ok (paywall·빈 본문·JS 필수 페이지는 근거 제외) |
| 인용문 | `quote` 가 해당 block(`page_or_locator`)의 실제 텍스트에 포함, `content_hash` 일치, 20단어·240자 이내 |
| 출처 | 검색 응답이 실제로 인용한 URL 만 허용 |
| 날짜 | 기준일 이후 자료 제외, 발행일이 원문·메타데이터에 없으면 None 으로 낮추고 한계 기록 |
| 수치·단위 | 요약문의 숫자·단위가 인용 block 에 있어야 함 |
| 입장 일치 | positive↔support, negative↔counter, 조건부 counter 는 구체적 조건 필요 |
| 범위 | 데이터센터 관련성, 선정 기술 직접 발언(`selected_technology`)과 기술 계열 배경 의견(`technology_family`) 구분 |

실패 항목은 근거가 되지 않고 `gaps`/`limitations` 에 사유와 함께 남는다. 반대 의견을 **찾지 못한 경우**는 가짜 근거를 만들지 않고
`not_found`(검색은 완료됐으나 확인되지 않음)로 기록하며, `blocked`(검색·검증 미완료)·`unsearched`(예산 부족)와 구분한다.

## 부모 State 와의 계약 (`node.py`)
- 읽는 키: `request`(as_of, max_search_rounds), `selected_tech`, `domain`(`datacenter` 포함), `technical_findings`(주장 문장만 맥락으로 사용), `rework_hint`(Supervisor 가 주입)
- 쓰는 키: `stakeholder_findings`(PerspectiveFindings), `evidence_store`(새 근거만), `search_log_by_perspective`, `quality_by_perspective["stakeholder"]`
- 다른 관점의 결과는 읽지 않는다.
- `records` 는 기술 × 그룹마다 하나(지지/반대/중립 건수, 직접/기술군 scope). 찾지 못한 조합은 `gaps` 의 `not_found: …` 로 남는다.

## Supervisor 와의 관계
재작업 요청(`rework_hint`)이 오면 `gaps` 에 있는 (기술, 그룹) 조합만, `focus_queries` 를 덧붙여 다시 검색한다.
라운드 수는 워커 래퍼(`graph/workers.py`)가 `request.max_search_rounds` 에 이미 더해 준다. 충분성 기준(근거 3건 이상·출처 2곳 이상)과
재작업 예산(관점당 2회)은 `graph/sufficiency.py` 에 있다.

## 한계
- 검증은 문자열 포함·해시·날짜·수치 존재까지다. 발언자 신원, 번역 정확성, 인과·비교 기준의 타당성은 코드가 증명하지 못한다(중립성 Judge·사람 검토 필요).
- 현재 fetcher 는 공개 HTML 만 지원한다(PDF·JS 렌더링 필수 페이지 제외).
- 실제 API 로 end-to-end 실행한 기록은 아직 없다(오프라인 테스트 20개만 통과).
