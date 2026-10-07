"""팀 공유 보고서 규칙. 보고서 검증(agents/report)과 품질 평가(agents/quality)가 같은 표를 쓴다.

규칙을 바꿀 때는 이 파일만 고친다. 코드 로직과 섞이지 않게 값(표)만 둔다.
"""

from __future__ import annotations


# ── 중립성: 우열·추천 표현 ─────────────────────────────────────────────
# 기술 평가 목적은 우열 판정이 아니다(과제 가이드 "중립성"). 보고서 검증과 품질 평가가 같은 목록을 쓴다.
PROHIBITED_COMPARISON: tuple[str, ...] = (
    "승자", "추천", "권장", "압도", "월등", "우위", "열위",
    "더 낫", "우수하", "최고의", "최선의", "선택해야", "바람직하",
)

# "원본 확인 권장"처럼 검증 절차를 권하는 표현은 기술 추천이 아니다(live 1차 오탐).
VERIFICATION_ADVICE_PATTERN = r"(확인|검토|검증|재현)\s*(을|를)?\s*권장"


# ── Groundedness: 핵심 수치의 측정 조건 ─────────────────────────────────
# 선정 기술(DeepSeek-V2 MLA, ITME) 원논문의 수치는 측정 조건과 함께 써야 한다(RAG 단계 팀 규칙).
# 도메인 규칙이라 선정 기술이 바뀌면 이 표만 교체한다.
#   metric       : 수치 정규식
#   required     : 문장(공백 제거·소문자)에 모두 있어야 하는 조건
#   forbidden    : 있으면 안 되는 귀속(예: MLA 단독 효과)
#   condition    : 조건이 빠졌을 때 덧붙일 표준 문구(None이면 자동 보정 불가)
#   evidence_key : 표준 문구를 덧붙여도 되는지 확인할 근거 원문 키워드. 근거에 없으면 덧붙이지 않는다.
METRIC_RULE_SPECS: tuple[dict, ...] = (
    {
        "code": "metric_93_3_context",
        "metric": r"93\.3",
        "required": (r"deepseek67b",),
        "forbidden": (r"mla만으로", r"mla단독"),
        "message": "93.3%에는 DeepSeek 67B 대비 전체 비교 조건이 필요함",
        "condition": "DeepSeek 67B 대비 DeepSeek-V2 전체 모델 비교",
        "evidence_key": r"deepseek67b",
    },
    {
        "code": "metric_5_76_context",
        "metric": r"5\.76",
        "required": (r"8[×x]h800|8개h800",),
        "forbidden": (),
        "message": "5.76×에는 8×H800 조건이 필요함",
        "condition": "8×H800 기준",
        "evidence_key": r"h800",
    },
    {
        "code": "metric_35_7_context",
        "metric": r"35\.7",
        "required": (r"최대",),
        "forbidden": (),
        "message": "35.7%는 최대값으로 표시해야 함",
        "condition": "CPU-offload 대비 최대값",
        "evidence_key": r"cpu[-_]?offload",
    },
    {
        "code": "metric_1_81_context",
        "metric": r"1\.81",
        "required": (r"ttft", r"turn5", r"recomputation"),
        "forbidden": (),
        "message": "1.81×에는 turn 5, TTFT, recomputation baseline 조건이 필요함",
        "condition": "turn 5 TTFT, recomputation 대비",
        "evidence_key": r"recomput",
    },
    {
        "code": "metric_42_5_attribution",
        "metric": r"42\.5",
        "required": (r"훈련|training",),
        "forbidden": (r"mla",),
        "message": "42.5%를 MLA 효과로 귀속할 수 없음",
        "condition": None,
        "evidence_key": None,
    },
)
