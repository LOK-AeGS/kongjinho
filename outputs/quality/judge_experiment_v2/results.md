# Judge model experiment

각 모델·변형을 2회 평가했다. 가격은 코드의 1M token당 추정치이며 실제 청구액이 아니다.

## Per-model summary

| model | detection | false positive | consistency | mean score diff | latency(s) | tokens | est. USD | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-4.1-mini | 50.0% | 50.0% | 100.0% | 0.20 | 5.17 | 97866 | 0.0465 | 0 |
| gpt-4.1 | 100.0% | 50.0% | 100.0% | 0.00 | 5.76 | 100193 | 0.2510 | 0 |
| gpt-4o | 100.0% | 50.0% | 100.0% | 0.20 | 5.38 | 97027 | 0.2821 | 0 |
| gpt-5-mini | 100.0% | 50.0% | 100.0% | 0.40 | 53.73 | 140933 | 0.1214 | 0 |

## Per-variant detection matrix

clean 행은 오탐 없이 통과한 비율, 결함 행은 목표 기준을 탐지한 비율이다.

| model | variant | expected | code correct | LLM correct | hybrid correct |
|---|---|---|---:|---:|---:|
| gpt-4.1-mini | clean | no failure | yes | 0.0% | 0.0% |
| gpt-4.1-mini | neutrality_implicit | neutrality | no | 0.0% | 0.0% |
| gpt-4.1-mini | groundedness_fabricated | groundedness | yes | 100.0% | 100.0% |
| gpt-4.1-mini | bias_onesided | bias_control | yes | n/a | 100.0% |
| gpt-4.1-mini | coverage_missing | coverage | yes | n/a | 100.0% |
| gpt-4.1 | clean | no failure | yes | 0.0% | 0.0% |
| gpt-4.1 | neutrality_implicit | neutrality | no | 100.0% | 100.0% |
| gpt-4.1 | groundedness_fabricated | groundedness | yes | 100.0% | 100.0% |
| gpt-4.1 | bias_onesided | bias_control | yes | n/a | 100.0% |
| gpt-4.1 | coverage_missing | coverage | yes | n/a | 100.0% |
| gpt-4o | clean | no failure | yes | 0.0% | 0.0% |
| gpt-4o | neutrality_implicit | neutrality | no | 100.0% | 100.0% |
| gpt-4o | groundedness_fabricated | groundedness | yes | 100.0% | 100.0% |
| gpt-4o | bias_onesided | bias_control | yes | n/a | 100.0% |
| gpt-4o | coverage_missing | coverage | yes | n/a | 100.0% |
| gpt-5-mini | clean | no failure | yes | 0.0% | 0.0% |
| gpt-5-mini | neutrality_implicit | neutrality | no | 100.0% | 100.0% |
| gpt-5-mini | groundedness_fabricated | groundedness | yes | 100.0% | 100.0% |
| gpt-5-mini | bias_onesided | bias_control | yes | n/a | 100.0% |
| gpt-5-mini | coverage_missing | coverage | yes | n/a | 100.0% |
