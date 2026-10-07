# 실제 API 실행 기록 (`python main.py --live all --supervisor llm --debug`)

2026-10-07, 여섯 노드 모두 실제 API, Supervisor `gpt-5.5`. 결과 폴더는 `outputs/graph/<실행시각>/`(git 제외)이며, 이 문서는 그 요약이다.
trace_id: `trace-9ecc3ecc976e`. 전체 소요 약 18분 30초(병렬 구간 겹침 제외 대기 시간 약 1,100초).

## 결과
| 항목 | 값 |
|---|---|
| 종료 상태 | **ok** (supervisor 11턴, 12번의 결정 모두 `source=llm`, 폴백 0회) |
| 품질 평가 | 2회 — 1차 미달(bias·groundedness·length) → 도메인 재작업 → 보고서 재작성 → 2차 **통과** |
| 보고서 PDF | **10쪽**(표지 생략, 글자 94%) — 한도 이내 |
| 보고서 에이전트 자체 검사 | 위반 8건 (이전 실행 33건) — 남은 것은 수치 조건(93.3%·5.76×·1.81×) 문장과 "근거에 없는 측정값" |
| 근거 | 전체 76건 — technical 21 / market 10 / stakeholder 6 / domain 39 |
| 관점 상태 | 네 관점 모두 `partial`(검색 한도 안에서 확인하지 못한 공백이 남음), 한계점 절에 명시 |

## Supervisor 경로 (결정 로그 요약)
`technical → [market + stakeholder + domain] 병렬 → stakeholder 재작업(근거 2건 < 3건) → synthesis → report → quality(미달: bias·groundedness·length) → domain 재작업 → synthesis → report(품질 사유를 받아 재작성) → quality(통과) → finish`

## 노드별 시간 (초)
technical 141 · market 62 · stakeholder 121(재작업 183) · domain 122(재작업 151) · synthesis 20/17 · report 203/234 · supervisor 호출 평균 약 2.5.
병목은 보고서 생성(2회, 437초)이고 이해관계자는 병렬화 후 더 이상 병목이 아니다.

## 직전 실행(2026-10-07 14:48, 개선 전)과의 비교
| 항목 | 개선 전 | 이번 |
|---|---|---|
| 종료 상태 | degraded (품질 루프 상한에서 미달 종료) | **ok** (2차 평가 통과) |
| 보고서 자체 검사 위반 | 33건 | **8건** |
| groundedness | 청크·주장·문서 ID 인용으로 12건 미달 | 통과 |
| 재작업이 결과를 바꿨는가 | 아니오(위반 32→33) | 예(도메인 근거 16→39건, 보고서 위반 6→8로 수치 문장만 남음) |
| 이해관계자 | 근거 3건, 재작업이 결과를 덮어씀 | 근거 6건, 재작업 결과를 병합(주장 6개, 판정 4칸) |

## 남은 한계
- 이해관계자 근거는 6건·판정 4칸으로 6개 (기술, 그룹) 쌍을 모두 채우지 못했다(검색에서 확인되지 않은 쌍은 `not_found`/`blocked`로 기록).
- 보고서 에이전트 자체 검사에 수치 조건 8건이 남았다(`needs_review`). 부모 품질 평가는 통과했지만 두 검사가 보는 항목이 다르다.
- Judge(LLM)는 켜지 않았다(`--judge`). 임계값(3건·2곳·60%)은 임의로 정한 값이다.
- LangSmith 캡처는 계정에서 직접 찍어야 한다(`README` Usage 참고).
