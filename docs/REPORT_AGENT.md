# 보고서 생성 에이전트

보고서 생성 에이전트는 기술·시장·이해관계자·도메인 평가와 종합 결과를 LLM으로 서술하고 최종
Markdown으로 조립한다. 웹 검색과 RAG 검색은 수행하지 않으며, LLM에는 섹션별 최소 입력만 전달해
입력에 없는 사실·수치·사례를 새로 만들지 못하도록 제한한다. 인용·수치·REFERENCE 검증은 LLM이
아닌 결정적 코드가 담당한다.

## 부모 Graph 연결

보고서 노드는 확정된 `AppState`를 입력으로 사용한다. `node.py`는 부모 State에서 필요한 값을
내부 정규형으로 투영한 뒤 다음과 같이 보고서 에이전트 소유 키만 partial update로 반환한다.

- `report_sections`: 섹션별 Markdown과 `final_markdown`
- `references`: 실제 사용한 근거에서 만든 공통 `Reference`
- `quality_by_perspective["report"]`: 품질 상태, 위반, 경고, 검사한 claim ID
- `retries["report"]`: 보고서 섹션 수정에 사용한 누적 횟수
- `run_meta["report"]`: prompt version, 최종화 단계, PDF 출력 경로 기록

부모 State를 직접 수정하지 않는다. 부모 그래프에는 다음처럼 주입한다.

```python
from agents.report import make_node

# 기본값: OpenAI gpt-4o-mini로 섹션 작성
report_node = make_node()
```

저장소 루트 `.env`의 `OPENAI_API_KEY`를 자동으로 읽는다. 별도의 키 입력은 받지 않는다. 모델을
바꾸거나 이미 초기화한 LangChain client를 사용할 때는 State가 아니라 런타임 의존성으로 주입한다.

```python
from langchain.chat_models import init_chat_model
from agents.report import ReportAgentDeps, make_node

llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)
report_node = make_node(ReportAgentDeps(llm=llm, model="gpt-4o-mini"))
```

PDF까지 함께 만들 때는 런타임 의존성으로 출력 경로를 지정한다.

```python
from agents.report import ReportAgentDeps, make_node

report_node = make_node(
    ReportAgentDeps(pdf_output_path="output/pdf/technology-comparison-report.pdf")
)
```

## 입력

확정 AppState의 다음 키를 읽는다.

- `request.as_of`, `request.language`, `request.scope`
- `selected_tech`: SW·HW 기술명, 접근 방식, 선정 이유
- `domain`
- `technical_findings`, `market_findings`, `stakeholder_findings`, `domain_findings`
- `evidence_store`
- `synthesis`
- `retries["report"]`

공통 Claim의 `text`, `technology`, `perspective`, `limitations`와 공통 Evidence의 `id`, `doc_id`,
`author_or_org`, `published_at`, `page_or_locator`, `quote`를 보고서 내부 필드로 정규화한다.

## 출력

`report_sections`에는 다음 섹션과 최종 조립본이 저장된다.

```text
summary
background
technology_selection
technology_overview
trl
market
stakeholder
domain
comparison_matrix
conditions
conflicts
shared_and_complement
open_questions
limitations
reference
final_markdown
```

`references`는 공통 State의 `Reference` 형식을 그대로 따른다.

```python
{
    "evidence_id": str,
    "title": str,
    "author_or_org": str,
    "published_at": str | None,
    "url": str | None,
    "locator": str,
}
```

canonical 텍스트 산출물은 `report_sections["final_markdown"]`이다. `pdf_output_path`가 지정되면 같은
내용을 A4 PDF로 저장하고 절대 경로를 `run_meta["report"]["pdf_path"]`에 기록한다. 공통 State의
키는 추가하거나 변경하지 않는다.

## 실행 흐름

```text
입력 정규화
→ 목차에 맞춰 본문 섹션 작성
   └─ 섹션별 prompt + 최소 payload로 LLM structured output 호출
→ finalize_report()
   ├─ LLM으로 SUMMARY 작성
   ├─ claim → evidence ID 인용 연결
   ├─ 실제 사용 evidence로 REFERENCE 생성
   ├─ 섹션·전체 보고서 품질 검사
   └─ passed / repair / needs_review 판정
→ repair이면 위반 섹션과 검증 오류만 LLM에 전달해 부분 수정
→ finalize_report() 재실행
→ 통과하면 최종 Markdown 반환
→ PDF 출력 경로가 있으면 같은 내용으로 PDF 생성
→ 수정 한도 도달 시 needs_review와 미해결 항목 반환
```

사진의 `SUMMARY 작성 → 인용 연결 → REFERENCE 생성 → 품질 검사 → 통과?` 구간은
`subgraph.py::finalize_report()` 한 함수가 순서대로 수행한다. `run_report()`는 그 앞의 본문 작성과
통과 판정 이후의 부분 수정 분기만 담당한다.

확정 AppState에는 별도의 보고서 수정 한도 필드가 없으므로 보고서 설계의 최대 2회를 적용한다.
이미 `retries["report"]`가 1이면 한 번만 더 수정할 수 있고, 2이면 바로 검토 필요 상태로 종료한다.

## LLM 섹션 작성 방식

기본 writer는 `LLMSectionWriter`다. `ReportAgentDeps`에 주입된 LangChain chat model이 있으면 그것을
사용하고, 없으면 `model`, `model_provider`, `temperature` 설정으로 client를 초기화한다. 기본값은
OpenAI `gpt-4o-mini`, temperature 0이다.

각 섹션 호출에는 `SYSTEM_RULES`, 섹션 규칙, 정확한 제목, 해당 섹션의 최소 payload가 전달된다.
모델 출력은 `TypedDict` 기반 `LLMSectionOutput`으로 구조화하며 다음 세 필드만 받는다.

- `markdown`: 지정 제목으로 시작하는 섹션 본문
- `claim_ids`: 본문에서 실제 사용한 입력 claim ID
- `evidence_ids`: 본문에서 실제 인용한 입력 evidence ID

본문 검증이 실패하면 기존 초안과 해당 섹션의 검증 오류를 `build_repair_prompt()`에 추가하여
`repair()`를 호출한다. 전체 보고서를 다시 생성하지 않으며 수정 한도는 누적 2회다.

별도의 writer 구현이 필요하면 `ReportAgentDeps(writer=...)`로 다음 인터페이스를 주입할 수 있다.

```python
class Writer:
    def write(self, section_id, context): ...
    def repair(self, section_id, draft, issues, context): ...
```

LLM 및 외부 writer에는 해당 섹션의 `context["prompt"]`와 `context["payload"]`만 제공한다. 전체
evidence store를 한 번에 전달하지 않는다. 결정적 테스트 writer처럼 내부 정규형이 필요한 경우에만
`writer_receives_full_context=True`를 사용한다.

API 없이 재현하거나 테스트할 때만 다음처럼 결정적 모드를 명시한다.

```python
deps = ReportAgentDeps(generation_mode="deterministic")
```

## 인용 및 REFERENCE

1. 섹션이 사용한 claim ID에서 evidence ID를 수집한다.
2. 섹션 writer가 명시한 evidence ID와 합친다.
3. 누락된 인용 표시는 최종화 단계에서 연결한다.
4. 본문에서 실제 사용된 evidence만 REFERENCE 후보가 된다.
5. `doc_id → URL → metadata` 순서로 동일 source를 판별해 중복 제거한다.
6. 중복 제거한 출처를 `특허 → 논문 → 기타` 순서로 묶고 각 항목에 자료 유형을 표시한다.
7. 특허는 `저자(연도). 제목. 특허번호. URL`, 논문은 `저자(연도). 제목. 학술 저장소·식별자`,
   기타는 `저자(날짜). 제목. 발행 사이트. URL` 형식을 사용한다.
8. 공통 Evidence에 존재하는 메타데이터만 기록하며 없는 값은 추정하지 않는다.

```text
- 특허 : NVIDIA(2025). *KV Cache Transform Coding*. US-XXXXXXX-A1. https://...
- 논문 : Zandieh, A. et al.(2025). TurboQuant: Online Vector Quantization. *arXiv*, 2504.xxxxx.
- 기타 : Google Research(2026-03-30). *TurboQuant for KV Cache Compression*. Google Research Blog. https://...
```

## 결정적 품질 검사

- 필수 목차와 섹션 순서
- SUMMARY 첫 위치와 REFERENCE 마지막 위치
- claim/evidence ID 무결성
- 본문 인용과 섹션 evidence manifest 일치
- 실제 사용 evidence만 REFERENCE에 포함
- 동일 source reference 중복 제거
- claim·조건·인용문에 없는 측정값 차단
- 핵심 수치 조건
  - 93.3%: DeepSeek 67B 대비 전체 비교, MLA 단독 귀속 금지
  - 5.76×: 8×H800 조건
  - 35.7%: 최대값 표시
  - 1.81×: turn 5, TTFT, recomputation baseline
  - 42.5%: MLA 효과 귀속 금지
- 서열·권고 표현 차단
- `not_found`를 실제 부재로 일반화하는 문장 차단
- partial/failed upstream 상태를 한계점에 노출

## Fixture와 테스트

`tests/agents/report/fixtures/report_cases.json`은 확정 AppState와 같은 필드 구조를 사용하며 다음
케이스를 포함한다.

1. complete
2. evidence_gap
3. invalid_evidence
4. numeric_violation
5. partial_upstream
6. duplicate_reference

빠른 계약 테스트는 `generation_mode="deterministic"`과 mock LLM을 사용하므로 실제 API를 호출하지
않는다.

```bash
python -m unittest tests/agents/report/test_report_agent.py -v
```

실제 테스트 환경에서 LLM 연결까지 확인하는 통합 테스트는 `.env`의 키를 자동으로 읽는다. fixture의
시장성 섹션을 실제 모델이 작성하게 한 뒤 structured output, claim/evidence ID 범위와 결정적 품질
검사를 확인한다. 키가 있으면 API 비용이 발생한다.

```bash
python -m unittest tests.agents.report.test_report_llm_integration -v
```

모델은 기본 `gpt-4o-mini`이며 `REPORT_TEST_MODEL`로 변경할 수 있다. `.env`에 키가 없으면 이 통합
테스트만 skip되고 나머지 계약 테스트는 계속 실행된다.

실제 LLM으로 단독 fixture 실행 (`.env` 자동 로드, API 비용 발생):

```bash
python -m scripts.run_report --fixture
```

API 없이 fixture 재현:

```bash
python -m scripts.run_report --fixture --deterministic
```

실제 AppState JSON으로 Markdown 파일 생성:

```bash
python -m scripts.run_report \
  --state /absolute/path/to/state.json \
  --output /absolute/path/to/report.md \
  --model gpt-4o-mini
```

Report Agent만 실행하는 환경에서는 LLM 및 PDF 의존성을 설치한다.

```bash
pip install -r agents/report/requirements.txt
```

Markdown과 PDF를 함께 생성:

```bash
python -m scripts.run_report \
  --state /absolute/path/to/state.json \
  --output /absolute/path/to/report.md \
  --pdf-output /absolute/path/to/report.pdf
```

PDF는 한글 폰트를 자동 탐색한다. 기본 후보가 없는 실행 환경에서는 `REPORT_PDF_FONT`에 사용할
TTF/TTC 파일의 절대 경로를 지정한다.

## 설계 문서·Graph 그림과 실제 구현의 차이

### 1. Report 내부 흐름의 표현 방식

설계 그림은 `본문 작성 → SUMMARY → 인용 → REFERENCE → 품질 검사 → 통과?`를 각각 독립된
노드처럼 표현한다. 실제 구현에서는 이 단계를 부모 LangGraph 노드로 쪼개지 않는다. 보고서 작성은
부모 Graph에서 하나의 `report` 노드이며, 내부에서 본문을 작성한 뒤 `finalize_report()`가 SUMMARY부터
통과 판정까지 한 번에 실행한다. 이 방식은 중간 결과를 부모 State에 반복 기록하지 않고 보고서
소유 키를 한 번에 반환하기 위한 것이다.

### 2. 수정 루프의 범위

그림의 수정 화살표는 본문 작성 단계 전체로 돌아가는 것처럼 보이지만, 구현은 검증에 실패한
섹션만 `writer.repair()`로 교체한다. 정상 섹션은 다시 만들지 않는다. 본문이 수정되면 SUMMARY,
인용, REFERENCE, 품질 검사는 다시 실행해 최종 조립본과 사용 근거 목록을 동기화한다.

### 3. 부모 Graph Edge

Report Agent는 부모 Graph의 `synthesis → report → END` 구간에서 실행되는 단일 노드다. 보고서
내부의 SUMMARY·REFERENCE·검사 단계를 `graph/build.py`의 별도 Edge로 등록하지 않는다. 이 내부
단계들은 Report Agent 전용 구현이며 다른 에이전트가 읽거나 갱신할 공유 State가 아니기 때문이다.

### 4. State 출력 형태

설계 설명에서 보고서 객체 하나로 표현할 수 있지만 확정 AppState에는 `report` 키가 없다. 따라서
실제 노드는 `report_sections`, `references`, `quality_by_perspective`, `retries`, `run_meta`로 결과를
나눠 반환한다. 최종 Markdown은 `report_sections["final_markdown"]`에 저장한다.

### 5. 수정 한도

확정 `RequestSpec`에는 `max_revision_rounds`가 없고 `retries`에는 사용 횟수만 있다. 따라서 보고서
에이전트는 설계에서 정한 최대 2회를 내부 상수로 적용하고, 누적 사용 횟수를 `retries["report"]`에
되돌려준다.

### 6. PDF 출력

설계 그림은 통과 후 Markdown·PDF 출력을 하나의 결과처럼 표시한다. 실제 구현도 동일한 최종
Markdown에서 PDF를 만들지만, 파일 생성 여부와 경로는 `ReportAgentDeps.pdf_output_path`로 주입한다.
확정 AppState에 PDF 전용 키가 없으므로 경로는 새 최상위 키가 아니라 허용된
`run_meta["report"]["pdf_path"]`에 기록한다. PDF 생성은 품질 검사와 수정 루프가 끝난 뒤 한 번만
실행되며, 최종 Markdown과 PDF 본문의 불일치를 막는다. PDF 렌더러는 A4 레이아웃, 한글 폰트,
제목·본문·표·페이지 번호를 적용하고, 원문에 없는 내용을 생성하지 않는다.

### 7. LLM 호출

보고서 본문과 SUMMARY는 실제 `LLMSectionWriter`가 작성한다. 프롬프트 파일은 더 이상 확장 지점만
남긴 코드가 아니라 모든 LLM 작성·수정 호출에서 사용된다. client와 모델 설정은 직렬화 대상인
공통 State에 넣지 않고 `ReportAgentDeps`로 주입한다.

실행 메타데이터에는 `prompt_version`과 함께 `generation.mode`, `provider`, `model`, `temperature`를
기록한다. API가 없는 계약 테스트와 완전 재현 실행을 위해 `deterministic` 모드는 명시적으로만
지원한다.
