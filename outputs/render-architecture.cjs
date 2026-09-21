const fs = require('fs');
const path = require('path');
const { chromium } = require('/Users/seunghwan/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const out = path.join(__dirname, 'agent-architecture');
const diagrams = [
['01-overall','전체 아키텍처',`flowchart TD
 A["사용자 입력<br/>SW·HW 기술 각 1개와 선정 이유<br/>평가 도메인 · 조사 기준일"] --> B["① 기술 조사 에이전트"]
 B --> C["② 시장 평가 에이전트"]
 B --> D["③ 이해관계자 평가 에이전트"]
 B --> E["④ 도메인 평가 에이전트"]
 C --> F["동기화<br/>세 평가 결과 수집"]
 D --> F
 E --> F
 F --> G["⑤ 평가 종합 에이전트"]
 G --> H["⑥ 보고서 생성 에이전트"]
 H --> I["다관점 평가 보고서<br/>SUMMARY → 본문 → REFERENCE"]
 V[("공유 문서 검색 저장소")] -. "RAG" .-> B
 V -. "RAG" .-> C
 V -. "RAG" .-> E
 W["외부 웹 검색 도구"] -.-> C
 W -.-> D`],
['02-rag-preparation','공유 RAG 준비 과정',`flowchart LR
 A["선정 문서<br/>총 200페이지 이내"] --> B["PDF·문서 파싱<br/>표·본문·페이지 정보 보존"]
 B --> C["청크 분할<br/>문서 ID · 페이지 · URL 연결"]
 C --> D["오픈소스 임베딩"]
 D --> E[("벡터 인덱스")]
 C --> F[("키워드 인덱스")]
 E --> G["공유 검색 도구"]
 F --> G`],
['03-technical-research','기술 조사 에이전트 — RAG',`flowchart TD
 A["입력<br/>선정 기술 · 원문 문서 · 조사 기준일"] --> B["조사 질문 생성<br/>원리 · 적용 범위 · 성능 · 한계 · 성숙도"]
 B --> C["RAG 검색 도구<br/>기술별 원문 검색"]
 C --> D["근거 적합성 검사<br/>질문 관련성 · 페이지 · 실험 조건"]
 D --> E{"근거가 충분한가?"}
 E -- "부족 · 재시도 가능" --> F["질의 재작성<br/>다른 절·표 검색"]
 F --> C
 E -- "충분" --> G["기술별 구조화 추출"]
 E -- "재시도 한도 도달" --> H["근거 부족 항목 기록"]
 H --> G
 G --> I["비교 조건 정규화<br/>모델 · 문맥 길이 · HW · 배치 · 측정 지표"]
 I --> J["TRL 추정<br/>검증 환경·운용 근거 연결<br/>공개 정보 기반 추정 명시"]
 J --> K["출력: technical_findings<br/>SW·HW 개요 / 성능 조건 / 한계 / TRL / 근거"]`],
['04-market-evaluation','시장 평가 에이전트 — RAG + 웹 검색',`flowchart TD
 A["입력<br/>선정 기술 · 기술 조사 결과 · 조사 기준일"] --> B["시장 조사 계획<br/>수요 · 채택 · 제품화 · 생태계"]
 B --> C["RAG 검색<br/>수집된 산업·제품 문서"]
 B --> D["웹 검색·원문 열람<br/>공식 발표 · 도입 사례 · 산업 자료"]
 C --> E["근거 정리<br/>발행일 · 대상 시장 · 출처 · 주장"]
 D --> E
 E --> F["증거 수준 구분<br/>전망 / 제품 발표 / 실증 / 실제 운용"]
 F --> G{"주요 항목의 근거가 충분한가?"}
 G -- "부족 · 재시도 가능" --> H["보완 검색 질문 생성"]
 H --> D
 G -- "충분 또는 한도 도달" --> I["시장 관점 비교<br/>채택 동인 · 도입 비용 · 확산 장벽"]
 I --> J["출력: market_findings<br/>시장성 / 채택 현황 / 생태계 / 불확실성 / 근거"]`],
['05-stakeholder-evaluation','이해관계자 평가 에이전트 — 웹 검색',`flowchart TD
 A["입력<br/>선정 기술 · 기술 조사 결과"] --> B["이해관계자별 검색 계획"]
 B --> C["경쟁 기술 진영<br/>대응 기술 · 공개 입장"]
 B --> D["도입 기업·개발자<br/>사용 후기 · 이슈 · 운영 장벽"]
 B --> E["투자·산업 관계자<br/>투자 동향 · 분석 의견"]
 C --> F["웹 검색·원문 열람"]
 D --> F
 E --> F
 F --> G["발언 맥락 추출<br/>발언자 · 소속 · 날짜 · 대상 기술"]
 G --> H["분류<br/>긍정 / 부정 / 조건부 / 중립"]
 H --> I["편향 점검<br/>홍보성 · 이해관계 · 중복 인용 · 대표성"]
 I --> J{"관점이 누락되었는가?"}
 J -- "예 · 재시도 가능" --> K["누락·반대 관점 검색"]
 K --> F
 J -- "아니오 또는 한도 도달" --> L["출력: stakeholder_findings<br/>관계자별 입장 / 쟁점 / 근거 / 미확인 영역"]`],
['06-domain-evaluation','도메인 평가 에이전트 — RAG',`flowchart TD
 A["입력<br/>선택 도메인 · 기술 조사 결과"] --> B["도메인 요구사항 정의<br/>메모리 · 지연 · 처리량 · 전력 · 정확도 · 비용"]
 B --> C["평가 질문 생성<br/>각 기술이 요구사항을 충족하는 조건은?"]
 C --> D["RAG 검색<br/>기술 원문 · 벤치마크 · 도메인 자료"]
 D --> E["적용 조건 검토<br/>실험 환경과 목표 환경의 차이"]
 E --> F{"판단 근거가 충분한가?"}
 F -- "부족 · 재시도 가능" --> G["질의 보완"]
 G --> D
 F -- "충분 또는 한도 도달" --> H["조건별 적합성 분석<br/>기대 효과 · 제약 · 운영 부담"]
 H --> I["사실과 추론 분리<br/>직접 검증 / 조건부 추론 / 판단 보류"]
 I --> J["출력: domain_findings<br/>도메인별 적합 조건 / 제약 / 근거"]`],
['07-synthesis','평가 종합 에이전트 — 구조화 결과 기반',`flowchart TD
 A["입력<br/>기술·TRL / 시장 / 이해관계자 / 도메인 결과"] --> B["결과 완결성 검사<br/>필수 항목 · 근거 ID · 조사 시점"]
 B --> C["비교 매트릭스 생성<br/>기술 × 평가 관점"]
 C --> D["관점 간 관계 분석"]
 D --> E["일치<br/>여러 관점에서 같은 판단"]
 D --> F["상충<br/>채택 기대와 운영 제약 등"]
 D --> G["보완<br/>SW·HW 병행 가능 조건"]
 E --> H["근거 강도·불확실성 반영"]
 F --> H
 G --> H
 H --> I["중립성 검토<br/>일방적 우열 · 과장 · 근거 없는 결론 점검"]
 I --> J{"검토 통과?"}
 J -- "수정 필요" --> K["결론 표현·범위 수정"]
 K --> I
 J -- "통과 또는 수정 한도 도달" --> L["출력: synthesis<br/>비교표 / 상충 지점 / 조건부 시사점 / 한계"]`],
['08-report-generation','보고서 생성 에이전트 — 구조화 결과 기반',`flowchart TD
 A["입력<br/>선정 이유 · 관점별 결과 · 종합 결과 · 출처"] --> B["목차 구성<br/>SUMMARY → 배경 → 기술 선정 → 기술 개요<br/>→ 관점별 평가 → 시사점 → 한계 → REFERENCE"]
 B --> C["본문 작성<br/>비교표 · 상충 지점 · 조건부 해석"]
 C --> D["SUMMARY 작성<br/>핵심 평가 결과 · 반 페이지 이내"]
 D --> E["인용 연결<br/>주장 → 근거 ID → 원문"]
 E --> F["REFERENCE 생성<br/>실제로 인용한 자료만 포함"]
 F --> G["품질 검사<br/>근거 누락 · 수치 일치 · 중립성 · 목차"]
 G --> H{"통과?"}
 H -- "아니오 · 수정 가능" --> I["문장·인용 수정<br/>미확인 주장은 삭제 또는 한계 명시"]
 I --> C
 H -- "예" --> J["Markdown·PDF 출력"]
 H -- "수정 한도 도달" --> K["검토 필요 상태 반환<br/>미해결 항목 명시"]`]
];
(async () => {
 const browser = await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
 const page = await browser.newPage({viewport:{width:2200,height:1800},deviceScaleFactor:2});
 await page.setContent('<html><head><meta charset="utf-8"></head><body style="margin:0;background:white"></body></html>');
 await page.addScriptTag({path:'/private/tmp/kv-mermaid.min.js'});
 await page.evaluate(() => mermaid.initialize({startOnLoad:false,theme:'default',fontFamily:'Arial, Apple SD Gothic Neo, sans-serif',flowchart:{htmlLabels:false,useMaxWidth:false,curve:'basis'},securityLevel:'strict'}));
 for (const [name,title,source] of diagrams) {
   fs.writeFileSync(path.join(out,'mermaid',name+'.mmd'),source+'\n');
   const svg = await page.evaluate(async ({source}) => (await mermaid.render('diagram',source)).svg,{source});
   fs.writeFileSync(path.join(out,'svg',name+'.svg'),svg);
   await page.evaluate(({svg})=>document.body.innerHTML=svg,{svg});
   await page.locator('svg').screenshot({path:path.join(out,'png',name+'.png')});
   console.log(name+' rendered');
 }
 fs.writeFileSync(path.join(out,'README.md'),'# KV cache Multi-Agent 아키텍처\n\n앞선 답변의 다이어그램 8개입니다.\n\n- png/: 고해상도 이미지\n- svg/: 확대 가능한 벡터 이미지\n- mermaid/: 편집 가능한 Mermaid 원본\n\n'+diagrams.map(([name,title])=>'- '+name+': '+title).join('\n')+'\n');
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
