# Supervisor 동작 시나리오

`python -m scripts.run_scenarios` 가 만든 파일이다. 워커는 가짜지만 supervisor·충분성 판단·품질 평가·종료 보장은 실제 코드이고, State 만 달라서 경로가 달라진다.

| 시나리오 | supervisor 턴 | 관점 재작업 | 품질 평가 | 종료 상태 |
|---|---|---|---|---|
| 01-normal | 6 | 0회 | 1회 | ok |
| 02-rework | 10 | 1회 | 2회 | ok |
| 03-degraded | 8 | 2회 | 1회 | degraded |

## 01-normal

모든 관점의 근거가 충분하고 보고서가 한 번에 품질 평가를 통과

- trace_id: `trace-01-normal` (LangSmith 태그 `scenario-01-normal`)
- 워커 호출 수: technical 1, market 1, domain 1, stakeholder 1, synthesis 1, report 1

경로: dispatch(technical) → dispatch(market,stakeholder,domain) → synthesis(-) → report(-) → quality(-) → finish(-)

## 02-rework

이해관계자 근거 부족 → 그 관점만 재작업, 보고서가 중립성 미달 → 보고서만 재작업(품질 Loop)

- trace_id: `trace-02-rework` (LangSmith 태그 `scenario-02-rework`)
- 워커 호출 수: technical 1, market 1, domain 1, stakeholder 2, synthesis 2, report 2

경로: dispatch(technical) → dispatch(market,stakeholder,domain) → dispatch(stakeholder) → synthesis(-) → report(-) → quality(-) → quality_rework(report) → synthesis(-) → report(-) → quality(-) → finish(-)

## 03-degraded

이해관계자 근거가 계속 부족 → 재작업 예산(2회) 소진, 한계를 안고 보고서까지 진행해 degraded 로 종료

- trace_id: `trace-03-degraded` (LangSmith 태그 `scenario-03-degraded`)
- 워커 호출 수: technical 1, market 1, domain 1, stakeholder 3, synthesis 1, report 1

경로: dispatch(technical) → dispatch(market,stakeholder,domain) → dispatch(stakeholder) → dispatch(stakeholder) → synthesis(-) → report(-) → quality(-) → finish(-)

저하 사유:
- stakeholder: 근거 부족인 채 재작업 예산 소진 (근거 1건 < 3건, 출처 1곳 < 2곳)
- 품질 평가 미달인 채 루프 상한 도달: bias
