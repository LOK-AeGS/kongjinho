# 기술조사 에이전트 실행 안내

기술조사 에이전트는 데이터센터 KV cache 병목을 대상으로 다음 두 판정 단위만 평가한다.

- SW: DeepSeek-V2의 Multi-head Latent Attention(MLA)
- HW: ITME(Inference Tiered Memory Expansion) 시스템 프로토타입

입력은 `data/technical/default_input.json`과 코드의 `DEFAULT_REQUEST`, `DEFAULT_SELECTED_TECH`에 동일하게 고정돼 있다. 실행자가 임의의 기술명으로 바꾸면 노드가 실패 결과를 반환한다. 출력 필드 집합은 `data/technical/output_contract.json`과 `validate_output_contract()`가 고정한다.

## 실행 구조

```text
collect_and_check
  ├─ 고정 PDF: BM25 + BGE-M3 dense + RRF
  ├─ Tavily Search: 데이터센터 운영 후보 URL 탐색
  └─ Tavily Extract: 후보 원문 확보 및 snapshot/hash 생성
        ↓
extract_normalize_and_bind
  ├─ gpt-4.1-nano Structured Outputs
  └─ quote가 원문에 실제 포함된 경우에만 evidence_id 생성
        ↓
trl_gate_assess
  └─ 코드가 TRL 1~9 Gate를 순서대로 계산
        ↓
validate_and_finalize
  ├─ 수치·baseline·TRL 운영 근거 검사
  └─ 부모 계약의 technical_findings/evidence_store 반환
```

LLM은 TRL 숫자를 생성하지 않는다. LLM은 구현물 수준, 검증 환경, 활동, 실제 LLM/가속기/대표 workload 여부 같은 관측만 구조화한다. 코드가 직접 근거가 결합된 관측을 대상으로 Gate 1부터 연속 충족된 최고 단계만 채택한다.

API, GitHub 코드, 프레임워크 지원, 제품 판매는 단독으로 TRL 7 이상을 만들 수 없다. 제한된 검색에서 상용·운영 근거를 찾지 못한 경우에도 “상용되지 않았다”고 쓰지 않고 “공개 근거에서 확인하지 못했다”고 기록한다.

## 환경 준비와 실행

`.env.example`을 `.env`로 복사한 뒤 값을 로컬에서 입력한다. 키는 코드, 문서, Git 기록에 넣지 않는다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt pytest
python -m scripts.run_technical
```

기본 결과 파일은 `outputs/technical/latest.json`이다. 최상위 키는 항상 다음 두 개다.

```json
{
  "technical_findings": {},
  "evidence_store": {}
}
```

`technical_findings`에는 다섯 기술 기준, 기술별 TRL 레코드, claim, gap, 공개 검색·품질 검사 metadata가 들어간다. `evidence_store`에는 원문 quote와 locator가 검증된 근거만 들어간다.

## 테스트

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/agents/technical/test_technical_agent.py
```

단위 테스트는 API와 embedding 모델을 호출하지 않는다. 가짜 retriever, Tavily, analyzer를 주입해 다음 계약을 검사한다.

- 고정 입력 및 변조 거부
- 부모 반환 키와 결과 결정성
- 검색 최대 2라운드
- 원문 quote 결합과 결정적 evidence ID
- 연속 TRL Gate와 announcement의 운영 단계 상향 차단
- MLA·ITME 수치 조건
- Tavily Search 후 Extract 호출과 answer 미사용

실제 실행은 네트워크 비용과 모델 다운로드가 발생하므로 단위 테스트와 분리한다.
