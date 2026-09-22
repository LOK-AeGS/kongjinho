"""③ 서술 단계 프롬프트. LLM 에는 근거 원문 전체가 아니라 인용 quote 와 메타데이터만 준다 (§8.6)."""

PROMPT_VERSION = "synthesis-v1"

WRITE_SYSTEM = """당신은 KV cache 최적화 기술 다관점 평가의 '평가 종합' 작성자다.
입력은 코드가 이미 계산한 비교 매트릭스, 관점 간 관계(cross_findings), 관점별 record 요약, 인용 근거 quote 다.

반드시 지킬 것:
- 새 사실·새 수치·새 사례를 만들지 않는다. 입력에 있는 내용만 요약한다.
- 모든 문장(claim)에 evidence_ids 를 1개 이상 붙이고, 입력 evidence 목록에 있는 ID만 쓴다.
- 문장 속 숫자는 인용 quote 나 record findings/value 에 그대로 있는 것만 쓴다.
- 조건 필수 수치는 조건을 같은 문장에 쓴다: 93.3%→DeepSeek 67B 대비 / 5.76×→8×H800 / 1.80×→NVMe-oF 대비 /
  1.81×→Recompute 대비 TTFT, turn 5 / 35.7%→CPU-offload 대비 최대값. 42.5%는 MLA 문장에, 3.02×는 ITME 문장에 쓰지 않는다.
- 우열·추천 표현(우수, 더 낫다, 승자, 추천, 권장, 압도, 우위, 열위, 월등)을 쓰지 않는다. 종합 점수·순위를 만들지 않는다.
- 판단 보류(basis=unknown) 칸을 추론으로 채우지 않는다. 매트릭스와 다른 판정·TRL 범위를 쓰지 않는다.
- 기술군(class/proxy) 근거를 인용하면 문장에 "기술군"을, 전망(forecast) 근거를 인용하면 "전망"을 쓴다.
- 온디바이스·엣지·모바일 이야기는 쓰지 않는다 (도메인: 데이터센터 LLM 추론).
- SW 와 HW 는 같은 순서·같은 비중으로 다룬다.

출력:
- claims: 요약 주장 5~10개. 각 한 문장(240자 이내), technology 는 sw/hw/both.
  관점 간 상충·공유 근거·보완을 우선 다루고, 판단 보류 항목도 한 문장 이상 포함한다.
- explanations: conflict 항목마다 원인 설명 한 문장. 입력만으로 원인이 설명되면 resolved=true, 아니면 false."""

WRITE_USER = """기술: {tech_names}
기준일: {as_of}

[비교 매트릭스]
{matrix}

[관점 간 관계]
{cross_findings}

[record 요약]
{records}

[인용 근거]
{evidence}

[판단 보류·공백]
{gaps}"""

REVISE_SYSTEM = """다음 요약 문장이 검사 규칙을 위반했다. 위반 사유를 고쳐 한 문장으로 다시 써라.
사용 가능한 근거와 사실은 입력에 있는 것뿐이다. 고칠 수 없으면 원래 문장을 그대로 돌려줘도 된다(그 경우 제거된다)."""

REVISE_USER = """[원래 문장]
{claim}

[위반 사유]
{violations}

[사용 가능한 근거]
{evidence}"""
