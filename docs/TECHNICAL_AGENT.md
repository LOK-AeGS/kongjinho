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
  ├─ gpt-4.1 Structured Outputs
  └─ quote가 원문에 실제 포함된 경우에만 evidence_id 생성
     (인용한 candidate_id 가 틀려도 다른 후보에 글자 그대로 있으면 출처를 바로잡는다)
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

## 왜 모델과 PDF 파싱이 TRL 을 좌우하는가 (2026-09-22 측정)

TRL Gate 는 관측에 **직접 근거**가 결합되어야만 단계를 인정한다(`trl.py`의 `_direct`). 근거는
모델이 제시한 인용문이 원문에 글자 그대로 있을 때만 만들어진다. 따라서 인용 결합이 깨지면
관측이 아무리 정확해도 모든 Gate 가 L1 부터 탈락하고 `level=None`(공개 근거 부족)이 된다.

실제로 세 가지가 겹쳐 TRL 이 계속 비어 있었다.

| 원인 | 증상 | 조치 |
|---|---|---|
| `extract_text()` 기본 `x_tolerance=3` | 단어가 붙어 추출됨(`ITMEemployslayer-wiseprefetching`). ITME 논문 6p 기준 25자 초과 토큰 437개 | `PDF_X_TOLERANCE=1.5` (붙음 437 → 4개) |
| 2단 조판 페이지를 통째로 읽음 | 좌우 칼럼이 줄 단위로 섞임(`its limited physical ca- weights and KV caches via RDMA pacity forces`) | 줄 시작 x 좌표로 2단을 탐지해 칼럼별로 읽음. ITME·KIVI·InfiniGen·CXL-PNM 은 2단, DeepSeek·TurboQuant 는 1단으로 판정 |
| `gpt-4.1-nano` | 원문을 그대로 옮기지 못하고 바꿔 씀 | 기본 모델을 `gpt-4.1` 로 올림 |

고정 코퍼스에서 잰 인용 통과율과 결과는 다음과 같다.

| 조건 | 인용 통과 | 근거 | TRL sw / hw | 소요 |
|---|---|---|---|---|
| gpt-4.1-nano, 파싱 수정 전 | 0% (0/21) | 0 | None / None | — |
| gpt-4.1-mini, x_tolerance 수정 | 23% (12/53) | 6 | None / None | — |
| gpt-4.1, x_tolerance 수정 | 60% (42/70) | 12 | 3 / None | — |
| gpt-4.1, + 출처 교정 | 70% (54/77) | 19 | 6 / None | — |
| **gpt-4.1, + 칼럼 분리 (현재 기본값)** | **84~87%** | 23~30 | **3~5 / 4** | 약 130초 |
| gpt-5, 같은 조건 | 91% (81/89) | 34 | 3 / 4 | 323초 |

gpt-5 가 가장 정확하지만 한 번 실행에 323초가 걸려 기본값은 `gpt-4.1` 로 둔다. gpt-5 계열을
쓰려면 추론 토큰 때문에 출력 한도와 타임아웃을 함께 올려야 한다. 기본값(8000 토큰, 120초)으로는
응답이 잘리거나 타임아웃이 나면서 **추출이 통째로 비고 그래프는 계속 진행한다.**

```bash
OPENAI_MODEL=gpt-5 TECHNICAL_MAX_OUTPUT_TOKENS=32000 TECHNICAL_TIMEOUT_SECONDS=900 \
  python -m scripts.run_technical
```

남은 한계는 인용 통과율이 100% 가 아니라는 점이다(10~16% 는 모델이 원문을 바꿔 써 기각된다).
도메인 에이전트처럼 근거를 `E1`·`E2` 짧은 라벨로 고르게 하면 이 손실을 없앨 수 있다. 프롬프트와
출력 스키마를 함께 바꿔야 해서 이번 변경에는 넣지 않았다.

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
