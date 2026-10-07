# 보고서 품질 평가

## 목적

최종 보고서의 근거 추적성, 중립성, 편향 통제, 관점 완결성을 재현 가능한 코드 검사와 제한된 LLM 판정으로 확인한다. 코드가 구조적 결함을 우선 차단하고, LLM은 코드로 판별하기 어려운 암시적 추천과 수치 문장-인용 근거의 함의만 평가한다.

## Hybrid 역할 분담

| 기준 | 코드 검사 | LLM 작업 | 통과 규칙 |
|---|---|---|---|
| groundedness | 인용률 ≥ 0.8, 미존재 ID 없음, 무인용 측정값 없음, report blocking 위반 없음 | 인용된 측정값 문장과 evidence excerpt의 함의 판정 | code 통과 AND LLM 점수 ≥ 3 AND unsupported 0건 |
| neutrality | 명시적 금지 표현 탐지 | 보고서 전체에서 암시적 추천·선호·도입 조언 탐지 | code 통과 AND LLM 점수 ≥ 3 |
| bias_control | 전체 출처/기술 균형, 4.1~4.4 섹션별 출처 집중과 단일 기술 편중 | 없음 | code 통과 |
| coverage | 4.1~4.4 존재, 실질 내용 또는 사유 있는 판단 보류, 인용 존재 | 없음 | code 통과 |

함의 판정이 실패하고 인용 ID의 소유 관점이 하나로 확정되면 해당 관점을 `rework_targets`에 넣는다. `technical:`·`market:`·`domain:`은 prefix로, prefix 없는 stakeholder ID는 관점별 findings의 인용 관계로 소유자를 찾는다.

## 실험 설계

- 기준 보고서: 실제 live run State를 deterministic report writer로 다시 생성
- 변형: `clean`, `neutrality_implicit`, `groundedness_fabricated`, `bias_onesided`, `coverage_missing`
- 모델: gpt-4.1-mini, gpt-4.1, gpt-4o, gpt-5-mini
- 반복: 모델·변형별 2회
- 측정: 목표 결함 탐지, clean 오탐, 반복 일관성, 점수 차이, 지연, token, 추정 비용

## v1: whole-report judge

아래는 repeat 0의 `(groundedness, neutrality, bias_control, coverage)` 점수다. 코드 검사는 `coverage_missing`만 탐지했다. Whole-report LLM은 gpt-4.1-mini를 제외하고 암시적 neutrality만 안정적으로 탐지했으며 fabricated grounding, one-sided bias, missing coverage는 탐지하지 못했다.

| 모델 | clean | neutrality | fabricated | bias | coverage |
|---|---|---|---|---|---|
| gpt-4.1-mini | (5,5,4,4) | (5,5,4,4) | (5,5,4,4) | (5,5,4,4) | (5,5,4,4) |
| gpt-4.1 | (5,5,5,5) | (4,2,5,5) | (5,5,5,5) | (5,5,5,5) | (5,5,5,5) |
| gpt-4o | (5,5,4,5) | (5,1,4,5) | (5,5,4,5) | (5,4,4,5) | (5,5,4,3) |
| gpt-5-mini | (3,5,5,4) | (4,1,4,4) | (3,4,4,4) | (4,5,4,4) | (4,5,5,3) |

## v2: neutrality + 측정값 entailment

`outputs/quality/judge_experiment_v2/results.md` 결과다.

| 모델 | 탐지율 | clean 오탐 | 일관성 | 평균 점수 차이 | 평균 지연 | 총 token | 추정 총비용 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-4.1-mini | 50% | 50% | 100% | 0.20 | 5.17초 | 97,866 | $0.0465 |
| gpt-4.1 | 100% | 50% | 100% | 0.00 | 5.76초 | 100,193 | $0.2510 |
| gpt-4o | 100% | 50% | 100% | 0.20 | 5.38초 | 97,027 | $0.2821 |
| gpt-5-mini | 100% | 50% | 100% | 0.40 | 53.73초 | 140,933 | $0.1214 |

clean의 50% 표시는 무작위 오탐이 아니라 모든 8회 판정이 같은 두 문장을 지적한 결과였다.

- 실제 결함: `ITME는 NVMe-oF 대비 최대 1.80배...`가 `domain:ev:a6b7ddda9683`을 인용하지만 quote는 DeepSeek-V2 내용이며 ITME/1.80 근거가 없다.
- 절단 오탐: MLA 5.76배 문장의 quote에는 5.76이 존재하지만 선두 400자 뒤에 있었다. 측정값을 정규화한 뒤 첫 일치 지점 중심의 약 400자 window를 제공하도록 수정했다. 일치값이 없을 때만 선두 400자를 쓴다.

## 모델 선택

기본 judge는 `gpt-4.1`이다. v2에서 목표 결함 탐지율과 반복 일관성이 모두 100%, 평균 점수 차이는 0.00이었고, 평균 약 5.8초·호출당 약 $0.025로 gpt-4o보다 저렴하며 gpt-5-mini보다 훨씬 빨랐다. gpt-4.1-mini는 암시적 neutrality를 놓쳤다.

## 출처 집중 임계값 보정

13개 live report의 4.1~4.4 섹션을 측정한 정상 최대 출처 비율은 0.09~0.64였다. 도메인 섹션은 primary paper에 합리적으로 집중되어 0.44, 0.55, 0.62, 0.64가 관측됐다. 따라서 `>0.6`은 경고, `>0.8`은 실패로 두며 병리 사례와 `bias_onesided`의 1.0을 차단한다.

## 한계

- 결함 변형은 합성 편집이므로 실제 오류 분포를 완전히 대표하지 않는다.
- 단일 기준 보고서를 사용해 도메인·문체 다양성이 제한적이다.
- temperature 0에서도 LLM 판정은 모델 업데이트와 실행 환경에 따라 달라질 수 있다.
- 비용은 실험 스크립트의 추정 단가이며 실제 청구액과 다를 수 있다.
