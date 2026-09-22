# 도메인 평가 에이전트 설계

KV cache 최적화 기술 두 건(SW: DeepSeek-V2 MLA, HW: ITME CXL-Hybrid)이
**데이터센터** 환경에서 각 요구사항 축마다 어떤 조건에서 적합하다고 평가받는지 조사하는
에이전트입니다. 우열을 판정하지 않고, 관점에 따라 평가가 어떻게 갈리는지를 근거와 함께
기록합니다.

검색 대상은 팀이 미리 고른 문서 풀이 아니라 공신력 있는 웹 출처입니다.
논문·표준 문서·벤더 공식 자료·주요 기술 매체에서 근거를 모읍니다.

부모 그래프에는 팀 공통 `AppState`(`graph/state.py`) 하나로 연결됩니다. 코드는
`agents/domain/` 아래 `node.py`(State 변환), `subgraph.py`(검색·분석 루프),
`prompts.py`, `tools/`(검색·수집·색인·재현성 기록)로 나뉩니다.

## 평가 도메인을 데이터센터로 고정한 이유

KV cache 문제의 본질은 연산 병목이 아니라 메모리 병목입니다. 문맥이 길어질수록 저장해야 할
Key-Value가 선형으로 늘어 가속기의 HBM 용량을 소진하는데, 이 압박이 실제 비용으로 환산되는
곳이 데이터센터입니다. 온디바이스에서는 단일 사용자가 자기 메모리를 쓰고 끝나지만,
데이터센터에서는 한 요청이 붙잡은 KV cache가 곧바로 다른 사용자 몫의 HBM을 잠식하고
동시 처리 가능한 요청 수를 떨어뜨립니다.

두 번째 이유는 비교 가능성입니다. HW 기술인 ITME는 CXL 기반 disaggregated memory를
전제합니다. CXL은 랙 스케일 인터커넥트 규격이라 온디바이스나 엣지에는 존재하지 않고,
그 환경을 도메인으로 잡으면 HW 진영 기술은 평가 자체가 성립하지 않습니다. SW 압축과
HW 메모리 확장이 같은 무대에서 경쟁하는 환경은 데이터센터가 사실상 유일합니다.

세 번째는 평가 지표가 정량적으로 존재한다는 점입니다. 처리량(tokens/s), 동시 요청 수,
HBM 점유량, 랙 전력 예산, 토큰당 비용처럼 합의된 지표가 있어 "좋다/나쁘다"는 주관 판단을
조건부 판단으로 환원할 수 있습니다.

다만 이 선택에는 한계가 있습니다. 데이터센터로 고정하면 "온디바이스에서는 MLA가 더 유리하다"
같은 반대 방향 평가를 놓칩니다. 이 한계는 `domain_findings.gaps`와 보고서 한계점 장에
남깁니다.

## 임베딩이 필요한가 — 측정으로 답한 부분

웹 검색만 쓸 거라면 임베딩은 필요 없다고 보는 게 자연스럽습니다. 검색 API가 이미 랭킹된
결과를 돌려주니 벡터 인덱스를 또 만들 이유가 없어 보입니다. 그런데 두 가지가 걸렸습니다.

첫째, 스니펫에는 조건이 없습니다. "처리량 1.80배 향상"은 스니펫에 나오지만 그 수치가 어떤
모델 크기·문맥 길이·배치에서 측정됐는지는 본문에만 있습니다. 도메인 적합성은 실험 환경과
목표 환경의 차이를 봐야 하는데, 조건을 못 읽으면 판단할 근거가 없습니다. 그래서 본문을
가져오게 했고, arXiv 논문은 30~50페이지라 통째로 프롬프트에 넣을 수 없어 골라내야 합니다.

둘째, 언어 문제입니다. 이 에이전트는 평가 질문을 한국어로 생성하는데 근거 문서는 영어입니다.
키워드 검색으로 이게 되는지 직접 재봤습니다.

| 방식 | 한국어 질의 Hit@5 | 영어 질의 Hit@5 |
|---|---|---|
| BM25 | 0.29 (2/7) | 1.00 (9/9) |
| dense (BAAI/bge-m3) | 0.86 (6/7) | 0.89 (8/9) |

BM25는 영어 질의에서는 완벽하지만 한국어 질의에서는 무너집니다. 한국어 질의와 영어 문서
사이에 겹치는 어휘가 없으니 당연한 결과입니다. 반대로 dense는 언어에 관계없이 고르게
동작합니다. 즉 임베딩은 "있으면 좋은 것"이 아니라 교차언어 검색을 성립시키는 필수 요소입니다.

동시에 영어 질의 구간에서는 BM25가 dense보다 낫기 때문에 둘을 함께 씁니다.
이 측정 결과를 반영해 질문 생성 프롬프트에 "절반 이상을 영어로 쓰라"는 지시를 넣었습니다.

## 검색 방식 비교 (전체 Golden Set 16문항, 청크 213개)

| 방식 | Hit@1 | Hit@3 | Hit@5 | MRR | 평균 지연 | 피크 메모리 |
|---|---|---|---|---|---|---|
| BM25 | 0.562 | 0.625 | 0.688 | 0.606 | 0.2ms | 0.0MB |
| dense (bge-m3) | 0.875 | 0.875 | 0.875 | 0.875 | 35.3ms | 0.1MB |
| hybrid (RRF) | 0.812 | 0.875 | 0.875 | 0.833 | 33.7ms | 0.1MB |

재현: `python -m agents.domain.tools.ablation --cache data/fetch_cache --embedding BAAI/bge-m3`

하이브리드가 dense 단독보다 MRR이 낮습니다(0.833 대 0.875). RRF 융합이 BM25의 약한 상위
결과를 끌어올리면서 1순위 정확도를 깎은 것으로 보입니다. 그럼에도 hybrid를 유지한 이유는
질의 언어가 섞이기 때문입니다. 영어 질의만 놓고 보면 BM25가 1.00으로 dense(0.89)보다 높아,
한쪽을 버리면 그 구간이 손해입니다.

## 임베딩 모델 선정

과제 요구사항에 따라 오픈소스 임베딩만 사용합니다.

| 후보 | 라이선스 | 최대 입력 | 검토 결과 |
|---|---|---|---|
| BAAI/bge-m3 | MIT | 8192 토큰 | **선정** |
| intfloat/multilingual-e5-large | MIT | 512 토큰 | 논문 청크가 자주 잘림 |
| nlpai-lab/KURE-v1 | MIT | 512 토큰 | 한국어 특화, 영어 논문 검색에 불리 |
| sentence-transformers/all-MiniLM-L6-v2 | Apache-2.0 | 256 토큰 | 영어 전용, 한국어 질의 불가 |

## 출처 정책

검색 제공자의 도메인 필터는 신뢰하지 않습니다. Tavily의 `include_domains`에 13개 도메인을
지정했더니 medium·substack·youtube가 그대로 반환되는 것을 확인했습니다(2개일 때는 정상
동작). 그래서 결과를 받은 뒤 `agents/domain/tools/websearch.py`의 등급표로 다시 거르고,
거른 이유를 검색 로그에 남깁니다.

등급은 paper, patent, standard, vendor, news 다섯이며 등급표에 없는 도메인은 채택하지
않습니다. 도메인 접미사만 보고 등급을 매겼더니 `forums.developer.nvidia.com`이
`nvidia.com`으로 끝난다는 이유로 vendor 공식 자료로 분류된 적이 있어, `forums.`,
`community.` 같은 서브도메인을 먼저 걸러냅니다.

## PDF 파싱에서 걸린 문제

pdfplumber 기본 설정으로 arXiv 논문을 추출하니 단어 사이 공백이 사라졌습니다.
`Inthepastfewyears,LargeLanguageModels` 같은 식이라 BM25 토크나이저가 통째로 한 토큰으로
잡습니다. `x_tolerance`를 기본값 3에서 1.5로 낮추니 정상 분리됐고 과분할도 나타나지
않았습니다(25자 초과 토큰이 페이지당 0~2개, 대부분 수식 기호).

## Evidence와 State

부모 State는 팀 공통 `AppState`(`graph/state.py`) 하나입니다. 도입 초기에는 팀 스키마가
확정되기 전이라 `agents/domain/state.py`에 로컬 확장을 따로 뒀지만, 팀이 `graph/state.py`를
`AppState`로 통일하면서(ISSUE.md 1-1) 그 파일은 삭제하고 `node.py`가 직접 변환합니다.

`evidence_store`는 리스트가 아니라 `id`를 키로 하는 dict입니다(`AppState.evidence_store`,
`merge_evidence_store` 리듀서). 리스트에 `operator.add`로 쌓으면 재시도할 때마다 같은
근거가 중복으로 들어가고, 여러 관점이 같은 출처를 찾았을 때 같은 자료가 여러 벌 남습니다.

에이전트 내부(`agents/domain/tools/evidence.py`)에서 쓰는 evidence_id는 출처 신원에서
결정적으로 유도합니다. 정규화한 URL, locator, 정규화한 인용문을 이어 SHA-256으로 해시합니다.
`node.py`가 이 내부 형식을 팀 `Evidence` 형식(`id`, `page_or_locator`, `primary_or_secondary`,
`direct_or_proxy`, `stance`, `evidence_level` 등)으로 변환합니다. paper/patent/standard
출처는 `primary`/`direct`로, 나머지는 `secondary`/`proxy`로 분류합니다.

관점별 결과는 별도 키로 분리합니다. `domain_findings`, `quality_by_perspective["domain"]`,
`search_log_by_perspective["domain"]`이며, 병렬 분기에서 다른 관점과 충돌하지 않습니다.
AppState에는 실행 오류·수집 페이지 수를 담을 전용 최상위 필드가 없어서 `run_meta["domain"]`에
남깁니다. 서브그래프에 넘기는 입력도 `project_input()`으로 추려, 시장·이해관계자 관점의
중간 결론이 도메인 판단에 새어 들어가지 않게 했습니다.

## 판정을 요구사항 축 단위로 세분화

팀 `VerdictRecord` 스키마가 `criterion` 필드를 갖고 있어, 이제 "기술 하나에 대한 뭉뚱그린
판정 1건"이 아니라 **요구사항 축 하나 × 기술 하나마다 레코드를 하나씩** 만듭니다. 6개 축 ×
2개 기술 = 최대 12개 레코드입니다. 실제 실행에서도 매번 12건이 정확히 생성됩니다. 근거가
전혀 없는 조합도 `assessment=unknown` 레코드로 남겨, 그 축을 놓쳤는지 판단 자체를 안 했는지
구분되게 합니다.

## 품질 검증: 결정적 코드 3종 → 단일 프롬프트로 통합 (v3)

처음에는 검증을 두 층으로 나눴습니다. 결정적 `guard`(citation, locator, quote, numeric
unit, date, reference integrity), 표현 `linter`(승자·추천·압도 표현 차단), 별도 LLM
`judge`(coverage·neutrality)로, 코드로 답할 수 있는 것을 확률적 판정자에게 맡기면 같은
입력에 다른 결과가 나온다는 게 이유였습니다.

이후 팀 논의로 이 구조를 단순화했습니다 — `analyze` 노드 하나가 판정·주장·자체 품질 점검
(`self_check`)을 한 번의 구조화 출력으로 내도록 바꿨습니다. 코드에 남긴 것은 참조 무결성
(존재하지 않는 근거 라벨을 조용히 제거) 하나뿐입니다. 이건 품질 판단이 아니라 없으면
`evidence_store` 조회에서 `KeyError`로 죽는 방어적 처리라서 남겼습니다.

### 트레이드오프를 실행으로 확인했다

이 결정에는 실제 대가가 있었습니다. 코드 guard가 있었을 때 실측으로 잡았던 오류 유형
(근거 없는 인용, 제품명 숫자를 측정값으로 착각하는 것)이 되돌아올 수 있다는 우려가
있었고, 세 번의 실행에서 실제로 다른 종류의 문제가 나타났다 사라졌다 했습니다.

**1차 실행(v3)**: 근거 하나가 경쟁 제품("Kimi K3")에 대한 SemiAnalysis 뉴스레터였는데,
이걸 HW(ITME) 판정에 유추 근거로 썼습니다. `basis=inferred`, `scope=class`로 신뢰도는
낮췄지만 `findings` 문장 자체에는 "Kimi K3"라는 실제 대상이 드러나지 않아, 그 문장만 보면
ITME 자체 근거처럼 읽혔습니다. `self_check.violations`는 0건이었습니다 — 즉 자체 점검이
이 문제를 놓쳤습니다.

**2차 실행(v3.1)**: 프롬프트에 "근거가 평가 대상이 아닌 다른 제품이면 violations에
'대상 불일치'로 적고 findings에도 실제 대상을 명시하라"는 규칙을 추가했습니다. 결과 —
Kimi K3 오염 자체는 사라졌지만, 이번엔 정상적인 근거(DeepSeek-V2가 DeepSeek 67B 설정을
따른다는 배경 설명 등)에도 "대상 불일치"가 **claim 12건 중 10건**에 과잉 발동했습니다.
규칙이 "이 근거가 대상 기술의 완전한 직접 실측인가"라는 기준으로 너무 엄격하게 읽힌
것입니다.

**3차 실행(v3.2)**: "같은 회사의 이전 모델·같은 아키텍처 계열을 배경으로 인용한 것은
불일치가 아니다. 완전히 다른 회사의 다른 제품 얘기를 그 기술의 실측치처럼 쓴 경우만
불일치다"로 규칙을 구체화했습니다. 결과 — `violations` 10건 → **3건**으로 줄었고, 남은
3건을 직접 검증한 결과 전부 실제로 타당한 지적이었습니다(ITME가 아니라 HBM 세대 일반논의,
PIM이라는 다른 기술을 다룬 근거). 그리고 Kimi K3를 인용한 주장은 이번엔 text 자체에
`"for Kimi K3"`라고 정직하게 명시했습니다 — 1차의 문제가 실제로 개선됐습니다.

세 번의 실행에서 배운 것은 "완전히 놓침 → 과잉 반응 → 적절히 발견"이라는 궤적이 코드
검증과 달리 프롬프트 문구 하나로 크게 흔들린다는 점입니다. 결정적 코드는 느슨하든
엄격하든 실행마다 같은 기준으로 판정하지만, 단일 프롬프트 자체 점검은 규칙 문구의 미묘한
차이에 훨씬 민감했습니다. 이 트레이드오프는 팀이 이미 인지하고 감수하기로 한 것이며,
`quality_by_perspective["domain"]`의 `violations`/`warnings`는 실행마다 반드시 확인해야
합니다.

## 종합 평가 에이전트로 넘기는 데이터

평가 결과를 긴 산문으로 넘기면 종합 단계에서 희석됩니다. 그래서 구조화 레코드로 넘기며
세 가지를 강제합니다. 주장은 한 건당 한 문장으로 제한합니다(240자). 판단은 열거형으로
고정합니다 — `assessment`는 suitable·conditional·unsuitable·unknown, `basis`는 팀 공통
`direct`·`inferred`·`unknown`·`not_applicable` 중 하나입니다. 근거 본문은 반복해 싣지
않고 ID로 참조합니다.

참조 구조에서 한 번 크게 틀렸던 부분을 남겨 둡니다. 최종 `claim_id`는 검증 단계가
부여하므로 모델이 미리 알 수 없어, 순번(1, 2, 3…)으로 판정을 연결하게 했더니 모델이 자리를
세다 어긋나 SW 판정이 HW 주장을 가리켰습니다. 참조된 ID 자체는 유효해서 "존재하는 ID인가"만
보던 검사가 통과시켰습니다. 지금은 모델이 `claim_key`를 직접 붙이고, 판정의
`technology_id`와 주장의 `technology_ids`가 어긋나면 연결을 끊습니다. 같은 이유로
`evidence_id`(해시)도 직접 인용하게 했더니 주장 전부가 근거를 인용하지 못했던 적이 있어,
`E1`/`E2` 같은 짧은 라벨로 보여주고 검증 단계에서 실제 ID로 환산합니다.

## 확증편향 방지

검색 질문은 기대 효과를 묻는 것과 제약·실패 조건을 묻는 것을 짝으로 만듭니다. 근거가
부족하면 추정으로 메우지 않고 `unknown`과 `gaps`로 남깁니다. 실험 환경과 데이터센터 환경의
차이는 `conditions`에 따로 기록해, 조건이 다른 비교가 종합 단계에서 이루어지지 않게 합니다.

## 재현성

`agents/domain/tools/manifest.py`가 실행마다 LLM provider·모델 ID(`gpt-4o`)·temperature·
프롬프트 버전, 임베딩 모델·device·precision·batch size·차원, OS·CPU·RAM·torch 백엔드,
패키지 버전, 산출물 SHA-256, git commit을 기록합니다.

검색은 질의 단위로 `data/search_cache/`에, 수집한 본문은 `data/fetch_cache/`에
남습니다. `TAVILY_API_KEY`가 없으면 공유 캐시를 재생하는 오프라인 목 모드로
동작합니다. `scripts/run_domain.py`도 같은 공유 캐시를 쓰고, 결과 산출물만
`outputs/domain/<실행시각>/`에 분리해 저장합니다.

## 실행 결과 (2026-09-22, PROMPT_VERSION domain/v3.2, 모델 gpt-4o)

최종 노트북 재실행에서 14개 코드 셀이 모두 오류 없이 완료됐습니다.
검색 2라운드, 질의 12건, 195/200페이지로 근거 31건을 모아 주장 12건과
판정 12건(6축 × 2기술)을 만들었습니다. 출처는 paper 27 / vendor 4,
primary 27 / secondary 4입니다.

`domain_findings.status`는 `partial`이며 지연시간, 정확도 유지, 인프라 도입 비용
축을 Gap으로 남겼습니다. `quality_by_perspective["domain"].status`는
`needs_review`이고, `hw-accuracy-1`과 `sw-power-1`의 근거 부족 2건을
`violations`로 보고했습니다. `warnings`에는 SW/HW 서술 분량 불균형 1건이 남았습니다.

본문 수집은 ACM(`dl.acm.org`)·OpenReview·IEEE Xplore가 403(봇 차단)을 반환해 여러 건
실패했고, `deepseek.com`은 연결이 끊겼습니다. 그래프는 중단되지 않고 오류를
`run_meta["domain"].errors`에 기록한 뒤 계속 진행했습니다.

## 남은 위험

`quality_by_perspective`의 판정 일관성이 코드 검증보다 낮습니다. 위 네 번의 실행이
보여주듯, 같은 파이프라인이라도 프롬프트 문구와 그날의 검색 결과에 따라 `violations`
개수와 내용이 크게 달라집니다.

캐시를 삭제하거나 질의·검색 제공자 설정이 바뀌면 다른 문서가 들어올 수 있습니다.

하이브리드 융합 가중치를 조정하지 않았습니다. 측정에서는 dense 단독이 MRR 기준으로
앞섭니다. Golden Set이 16문항이라 한 문항이 지표의 0.06을 좌우합니다.

200페이지 한도는 런타임 누적으로 막으므로 어떤 페이지가 들어갈지 사전에 정할 수 없습니다.

평가 도메인을 데이터센터로 고정했으므로 "온디바이스에서는 MLA가 더 유리하다" 같은 반대
방향 평가는 담기지 않습니다. 모든 판단은 공개 정보에 기반한 추정입니다.

## 파일 구성

| 경로 | 역할 |
|---|---|
| `agents/domain/node.py` | AppState ↔ 도메인 내부 State 변환, 팀 스키마(Evidence/Claim/VerdictRecord/Gap/QualityReport) 매핑 |
| `agents/domain/subgraph.py` | 서브그래프(질문 생성→검색→분석), 근거 생성, 참조 무결성 |
| `agents/domain/prompts.py` | 프롬프트 템플릿, 도메인 프로파일, 프롬프트 버전 |
| `agents/domain/tools/evidence.py` | 결정적 evidence_id, 멱등 병합, 검색 로그 |
| `agents/domain/tools/websearch.py` | 출처 등급표, 캐시, 재시도·레이트리밋, 오프라인 목 |
| `agents/domain/tools/fetch.py` | 본문 수집·파싱, 짧은 문서/긴 문서 분기 |
| `agents/domain/tools/index.py` | 청킹, BM25·dense·hybrid, 임베딩 실행 정보 |
| `agents/domain/tools/ablation.py` | BM25·dense·hybrid 비교 |
| `agents/domain/tools/golden_set.py` | 표지 기반 정답 16문항 |
| `agents/domain/tools/manifest.py` | 실행 환경·모델·산출물 체크섬 기록 |
| `agents/domain/tools/artifacts.py` | 색인 산출물(청크·FAISS·BM25) 저장 |
| `scripts/run_domain.py` | 단독 실행 진입점 (`python -m scripts.run_domain`) |
| `tests/agents/domain/test_domain_agent.py` | 계약 테스트 (LLM·네트워크 없음) |

## 부모 그래프 연결

```python
from agents.domain import DomainAgentDeps, make_node
from agents.domain.tools.websearch import build_search_provider
from graph.build import build_graph
from langchain.chat_models import init_chat_model

deps = DomainAgentDeps(
    llm=init_chat_model("gpt-4o", model_provider="openai", temperature=0),
    search_provider=build_search_provider(Path("data/search_cache")),
    embedding_model="BAAI/bge-m3",
    fetch_cache_dir=Path("data/fetch_cache"),
)
app = build_graph(..., domain=make_node(deps), ...)
```
