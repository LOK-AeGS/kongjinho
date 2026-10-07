# 이해관계자 평가 에이전트 (`agents/stakeholder`)

이 구현은 teammate의 `feat/supervisor-pattern` 브랜치(commit `1d1d82a`)에서 포팅했다.

## 목적

선정 기술 2개(SW·HW)와 경쟁 기술 진영, 도입 기업·개발자, 투자·산업 관계자의 데이터센터 관점 공개 반응을 조사해 지지·반대·중립으로 기록한다. OpenAI `web_search`는 URL 발견에만 사용하고, 해당 URL의 원문 HTML을 별도로 가져와 검증한다.

## 원문 검증

기존 `agents/stakeholder_eval.py`는 LLM이 발언자·인용문·URL·날짜를 채운 뒤 원문 대조를 하지 않아 가짜 인용이 `evidence_store`에 들어갈 위험이 있었다. 새 에이전트는 다음 조건을 코드로 확인하고 통과한 항목만 근거로 인정한다.

| 검사 | 통과 조건 |
|---|---|
| 원문 접근 | fetch 상태가 `ok`이며 공개 HTML 본문이 있음 |
| 인용문 | 지정 locator의 실제 block에 quote가 포함되고 content hash가 일치함 |
| 출처 | 검색 결과가 실제로 발견한 URL임 |
| 날짜 | 기준일 이후가 아니며, 원문에서 확인되지 않으면 `None`으로 낮춤 |
| 수치·단위 | 요약에 쓴 수치와 단위가 인용 block에 존재함 |
| 입장 | positive/support, negative/counter 등의 방향이 일치함 |
| 범위 | 데이터센터 관련성이 있으며 선정 기술과 기술군 의견을 구분함 |

실패 항목은 근거로 쓰지 않고 `gaps`와 `limitations`에 남긴다. 검색했지만 확인하지 못한 반응은 `not_found`, 원문 접근·검증 미완료는 `blocked`, 질의 예산으로 조사하지 못한 조합은 `unsearched`로 구분한다.

## 부모 State 계약

- 입력: `request`, `selected_tech`, `domain`, `technical_findings`, 선택적 `rework_hint`
- 출력: `stakeholder_findings`, `evidence_store`, `search_log_by_perspective`, `quality_by_perspective["stakeholder"]`
- 다른 관점의 결론은 읽지 않으며 technical 주장 문장만 검색 맥락으로 사용한다.
- 재작업 시 worker가 directive의 `focus`를 `focus_queries`로, 현재 gap 중 일치 항목을 `gaps`로 변환한다.

## 한계

- 검증은 문자열 포함, hash, 날짜, 수치 존재까지이며 발언자 신원과 번역 정확성까지 증명하지 않는다.
- 공개 HTML만 지원하며 PDF나 JavaScript 렌더링 필수 페이지는 제외될 수 있다.
- 검색 예산과 단일 기준 보고서 때문에 `not_found`는 실제 의견 부재를 의미하지 않는다.
