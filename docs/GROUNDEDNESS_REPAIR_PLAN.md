# 최종 보고서 주장–근거 불일치 해결안

검토일: 2026-10-07. 기준 코드: `fa5caa4`. 기준 실행: `outputs/graph/20261007-171209-920622`.
이 문서는 확인한 원인과 제안이다. 아래 수정은 아직 구현하지 않았으며, 저장된 보고서·판정을 변경하지 않았다.

## 확인한 결과

- 최종 hybrid 품질 평가는 groundedness·bias_control 미달. groundedness의 인용률은 52/53(0.9811)이지만 Judge가 검사한 측정값 행 8개 중 7개를 unsupported로 판정했다.
- 인용 존재, 숫자 일치, 문장 전체의 함의 일치는 서로 다른 조건이다. 현재 결정적 검사는 첫 두 조건을 중심으로 검사한다.
- Judge의 unsupported에는 실제 근거 결함과 입력 범위 부족이 섞여 있다. Judge 통과를 위해 임계값을 낮추거나 이유 없이 사실을 지우는 접근은 피한다.

## 코드와 저장 결과로 재현한 원인

### 1. 제외한 수치가 record.findings를 통해 다시 유입됨

`agents/domain/node.py::_bind_numeric_claims`는 claim 본문과 record.value의 수치를 quote와 대조한다. record.value가 실패하면 None으로 바꾸지만 record.findings와 assessment/basis는 유지한다.

기준 실행에서 `domain:ev:fa178141ef2e`의 quote는 메모리 관리자·prefetch 구현 설명이다. 1.80과 35.7은 없지만 이를 인용한 record.findings에는 두 성능 수치가 남아 있다. value=None과 함께 “수치가 인용 근거 원문에서 확인되지 않아 제외”라는 limitations가 기록돼 있다.

`agents/report/validators.py::_citation_binding_issues`는 원문 excerpt에 연결된 claim과 record.findings를 더해서 검사한다. 따라서 제외한 수치가 다시 검증 근거로 인정된다.

API 없이 다음 최소 실험을 실행했다.

```python
line = '- ITME는 1.80배 TTFT와 최대 35.7% 처리량 개선을 보인다〔근거: domain:ev:fa178141ef2e〕'
_citation_binding_issues('domain', line, context)
# []: 기존 검사는 통과

quote_only = {**context, 'claims': {}, 'findings': {}}
_citation_binding_issues('domain', line, quote_only)
# numeric_citation_mismatch: 인용 근거에 없는 수치(1.80, 35.7)
```

### 2. 생성된 조건으로 원문 조건을 보충함

`agents/report/subgraph.py::_evidence_text`는 excerpt에 상위 claim.conditions를 더한다. `metrics.py::condition_supported`는 DeepSeek67B, H800 등의 키워드 존재로 표준 조건 주석을 붙일 수 있다고 판단한다.

상위 claim의 조건이 잘못돼도 원문에 있는 조건처럼 취급될 수 있다. 또한 H800 키워드 하나가 있다는 사실만으로 “8×H800에서 이 수치를 측정했다”는 연결이 입증되지는 않는다.

### 3. 비교 대상·효과 귀속이 숫자 검사에서 분리되지 않음

93.3%와 5.76의 저장된 quote는 DeepSeek 67B 대비 DeepSeek-V2의 전체 비교 결과를 서술한다. 보고서는 이를 “MLA는 … 개선한다”로 압축한다. 뒤에 전체 모델 비교라는 주석을 붙여도 문장의 주어·인과 귀속은 남는다.

`graph/rules.py`는 93.3에서 “MLA 단독/MLA만으로” 같은 일부 표현만 금지하며, 5.76에는 귀속 금지 규칙이 없다. 단순 정규식 확장만으로 일반적인 인과 귀속 오류를 해결하기는 어렵다.

### 4. Judge에 인용 일부만 전달됨

`agents/quality/judge.py::extract_entailment_items`는 행당 `ids[:2]`를 사용한다. SUMMARY 첫 행의 인용 ID는 3개지만 전달된 excerpt는 2개였다. 세 번째 ITME 근거는 빠졌다. 이는 ITME 내용이 excerpt에 없다는 판정에 영향을 줄 수 있으나, MLA 효과 귀속 문제까지 해결하지는 않는다.

한 quote에서 여러 측정값·조건이 떨어져 있으면 첫 일치 수치 주변 400자만 보여주는 방법도 근거를 누락할 수 있다. 최대 15개 행만 검사하고 일부 섹션·메타 문구는 제외하는 범위 제한도 있다.

### 5. upstream 재작업이 실패 문장을 직접 수정하지 않음

품질 노드는 근거 ID의 소유 관점으로 재작업 대상을 정한다. 하지만 인용 원문이 정상이고 보고서 편집 과정에서 귀속이 바뀐 오류까지 upstream 조사로 돌릴 수 있다. 기술·시장·도메인은 현재 부모의 실패 문장·인용 ID·수정 이유를 서브그래프 입력으로 받지 않는다.

## 제안하는 구현 순서

### 1단계: 잘못된 수치가 원문 검증을 우회하는 경로 차단

우선 대상: `agents/domain/node.py`, `agents/report/validators.py`, `agents/report/subgraph.py`, `agents/report/metrics.py`.

- record.value뿐 아니라 findings·조건에 포함된 측정값을 해당 record의 quote로 검증한다. 제거한 값은 gap에만 남기고 사실 서술에는 재사용하지 않는다.
- 숫자가 빠진다고 assessment/basis까지 자동으로 정당화되지는 않는다. 남은 원문이 정성 판정을 지지하면 inferred와 제한을 명시하고, 그렇지 않으면 unknown/판단 보류로 낮춘다.
- 보고서 수치 결합 검증에서 생성된 claim·record를 원문 근거로 사용하지 않는다. claim/record는 검사 대상 또는 연결 정보로만 사용한다.
- 조건 주석도 quote 또는 원문 위치가 확인된 별도 근거로만 보충한다. 키워드 하나가 전체 실험 조건을 입증하도록 하지 않는다.
- 비교표와 결정적 fallback도 같은 검증을 거쳐야 한다. 결정적 출력은 상위 오류를 그대로 복제할 수 있다.

이 단계는 데이터 누출을 막지만 문장 전체의 함의 검사를 대체하지 않는다.

### 2단계: 측정 사실을 구조화하고 보고서 문장을 분리

제안 계약(현재 State에 구현되지 않음):

```text
measurement_id, subject, metric, value, unit,
baseline, statistic(maximum/minimum 등), workload, hardware,
evidence_ids, locator, supporting_spans, support_status
```

- supporting_spans는 원문에 실제 존재하는 텍스트와 위치로 검증한다. LLM이 생성한 조건을 supporting_spans로 넣지 않는다.
- 모델 전체 결과와 구성요소 단독 실험을 subject에서 구분한다. MLA 단독 효과는 별도 비교 실험 근거가 있는 경우에만 쓴다.
- 사실을 “DeepSeek-V2 전체 모델은 DeepSeek 67B 대비 …로 보고됐다”처럼 근거의 대상에 맞춰 작성한다. hardware 등 조건은 이를 확인한 원문이 연결된 경우에만 함께 쓴다.
- 하나의 불릿에 SW 수치와 HW 메커니즘을 합치지 않고 각각의 인용을 붙인다. 표도 기술별 셀의 근거를 구분한다.
- 도메인 적용 판단은 보고된 사실과 분리한다. KV 감소가 HBM 압박 완화에 기여할 수 있다는 해석은 inferred로 표시하고, SLA 충족·production 실증 같은 추가 결론으로 확대하지 않는다.

ITME 1.80/35.7은 현재 domain quote로 출판하지 않는다. 고정 PDF의 해당 실험 부분에서 수치·baseline·turn/workload를 다시 추출하고 검증한 경우에만 복원한다. 같은 문서의 다른 위치라는 이유만으로 기존 quote를 정당화하지 않는다.

### 3단계: Judge 입력 누락과 오류 분류 개선

우선 대상: `agents/quality/judge.py`, `agents/quality/prompts.py`, `agents/quality/node.py`.

- 모든 인용 ID를 전달하되 입력이 길면 배치로 나눈다. IDs 목록에는 있는데 excerpt에는 없는 근거가 없도록 검증한다.
- 수치별·조건별 원문 span을 모으고 주변 문맥도 전달한다. 발췌 밖 정보 때문에 판정이 불확실하면 전체 quote/검증된 원문 문맥으로 한 번 더 확인한다.
- 지원 판정을 supported / contradicted / insufficient_excerpt / missing_source / attribution_mismatch로 구분한다. insufficient_excerpt를 통과로 취급하지 않으며, 실제 근거 결함과도 구분한다.
- unsupported_items에 section_id, claim_id, evidence_ids, reason, failure_type을 보존한다. 현재는 일부 이유가 합쳐진 문자열과 제한된 feedback으로 전달돼 수정에 필요한 정보가 줄어든다.
- 정성 사실과 인과 귀속도 검증 대상으로 확장한다. 전체 보고서의 대상 행 수·검사 행 수·생략 이유를 결과에 기록한다.
- 원문이 비교 조건을 명시하지 않았다는 제한 설명을 새로운 성능 주장과 구분한다. 최신 35.7% 상충 설명의 Judge 지적은 별도 확인이 필요하다.

### 4단계: 오류 발생 위치에 맞춰 수정 루프 연결

- 원문은 정상이고 보고서 문장만 잘못됐으면 해당 보고서 섹션을 재작성한다.
- upstream claim/record부터 잘못됐으면 소유 에이전트에 실패 문장·근거 ID·이유·수정 초점을 전달한다. 새 근거를 찾거나 unsupported claim을 제거하도록 한다.
- 관점이 바뀌면 해당 결과를 사용하는 종합·보고서를 갱신한다. 기술 요약 수정이 이미 완료된 시장·도메인 판단에 영향을 주는지도 의존성 정책에 포함한다.
- 근거 개수가 늘었다는 이유만으로 수정 성공을 판단하지 않는다. 실패 항목의 해소 여부와 근거 적합성을 재검사한다.
- 재작업 상한은 유지하고, 최종 미달이면 명시적인 needs_review로 종료한다.

## 검증 기준

1. 기존 record.findings의 1.80/35.7을 원문 근거로 삼던 재현 사례가 차단된다. 결정적 fallback에서도 되살아나지 않는다.
2. SUMMARY의 세 번째 근거와 400자 밖 조건을 검사에 전달하는 사례가 통과하고, 실제 존재하지 않는 근거·조건은 실패한다.
3. DeepSeek-V2 전체 결과를 MLA 단독 효과로 바꾼 문장은 실패하고, 원문 대상과 비교 기준을 보존한 문장은 통과한다.
4. ITME 수치를 다른 quote에 붙인 문장은 실패한다. 실험 원문 위치와 baseline을 보존한 인용은 통과한다.
5. 표 셀·정성 해석·조건 누락·상위 에이전트 오류·보고서 편집 오류를 각각 검사하고 올바른 수정 대상으로 라우팅한다.
6. 저장된 6개 State로 재생성해 결과를 비교한 뒤, live 생성과 hybrid Judge로 최종 확인한다. API 없는 재생성만으로 hybrid 품질 통과를 주장하지 않는다.

최종 성공 조건은 SUMMARY·REFERENCE, PDF 10장 이하, 인용/수치/함의 검사, bias_control·coverage·neutrality를 모두 만족하는 것이다. 현재 bias_control 미달(전체 출처 집중도 0.42 > 0.4)은 groundedness와 별개이므로 따로 해결해야 한다. 출처를 형식적으로 분산하려고 관련 없는 근거를 붙이지 않는다.

## 권장 우선순위

먼저 1단계와 Judge의 모든 인용 전달을 구현하고 저장된 State로 검증한다. 이어서 측정 사실의 귀속·조건을 정리하고 보고서 수정 피드백을 구조화한다. 전체 조사 재실행은 이 경로를 고친 뒤 진행해야 동일 오류의 반복과 API 비용을 줄일 수 있다.
