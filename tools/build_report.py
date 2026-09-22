"""보고서 마크다운 -> 데스크탑 HTML/PDF.

로컬 파일이라 웹폰트를 쓰지 않고 시스템 폰트 스택만 사용한다.
mermaid 는 PDF 인쇄 경로에서 렌더링되지 않으므로 인라인 SVG 로 대체한다.
"""

import re
import subprocess
from pathlib import Path

import markdown

SRC = Path("/Users/sanlee/Desktop/SKALA/코딩파일/Ai-service/Capstone_Ai_RAG/docs/REPORT_domain_section.md")
OUT_HTML = Path("/Users/sanlee/Desktop/도메인평가에이전트_보고서.html")
OUT_PDF = Path("/Users/sanlee/Desktop/도메인평가에이전트_보고서.pdf")

# --- 그래프 다이어그램 (mermaid 대체) ---------------------------------------
BOX = "fill:#fff;stroke:#33404d;stroke-width:1.2"
ACC = "fill:#eef4fb;stroke:#2f5d8a;stroke-width:1.3"
DEC = "fill:#fdf3e3;stroke:#a97316;stroke-width:1.3"
OKB = "fill:#eaf4ee;stroke:#2f7a4d;stroke-width:1.3"
TXT = 'font-family:"Apple SD Gothic Neo",sans-serif;font-size:12px;fill:#1c2530'
SUB = 'font-family:"Apple SD Gothic Neo",sans-serif;font-size:10.5px;fill:#5b6a78'


def box(x, y, w, h, style, title, sub=None, rx=6):
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" style="{style}"/>'
    ty = y + (h / 2 + 4 if sub is None else h / 2 - 4)
    out += f'<text x="{x + w / 2}" y="{ty}" text-anchor="middle" style="{TXT}">{title}</text>'
    if sub:
        out += f'<text x="{x + w / 2}" y="{ty + 15}" text-anchor="middle" style="{SUB}">{sub}</text>'
    return out


def arrow(x1, y1, x2, y2, label=None, dash=False):
    d = ' stroke-dasharray="4 3"' if dash else ""
    out = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#55636f" '
           f'stroke-width="1.2" marker-end="url(#ah)"{d}/>')
    if label:
        out += (f'<text x="{(x1 + x2) / 2 + 6}" y="{(y1 + y2) / 2 - 3}" style="{SUB}">{label}</text>')
    return out


def build_svg() -> str:
    W, H = 680, 672
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg">',
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
         'markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#55636f"/></marker></defs>']
    cx = 250
    s.append(box(cx - 110, 10, 220, 42, ACC, "질문 생성", "기대효과·제약 짝 / 절반 이상 영어"))
    s.append(arrow(cx, 52, cx, 78))
    s.append(box(cx - 110, 78, 220, 42, BOX, "웹 검색", "출처 등급 필터 + 질의 캐시"))
    s.append(arrow(cx, 120, cx, 146))
    s.append(box(cx - 110, 146, 220, 42, BOX, "본문 수집", "200페이지 예산 검사"))
    s.append(arrow(cx, 188, cx, 214))
    # 분기: 문서 길이
    s.append(f'<polygon points="{cx},214 {cx+92},244 {cx},274 {cx-92},244" style="{DEC}"/>')
    s.append(f'<text x="{cx}" y="248" text-anchor="middle" style="{TXT}">문서 길이</text>')
    s.append(arrow(cx - 92, 244, 118, 244, "짧음"))
    s.append(box(10, 222, 108, 44, BOX, "그대로 근거", "색인 없음"))
    s.append(arrow(cx + 92, 244, 562, 244, "긺"))
    s.append(box(562, 216, 112, 56, OKB, "청킹 + 색인", "BM25 + dense"))
    s.append(arrow(64, 266, 64, 316)); s.append(arrow(64, 316, cx - 112, 316))
    s.append(arrow(618, 272, 618, 316)); s.append(arrow(618, 316, cx + 112, 316))
    s.append(box(cx - 112, 296, 224, 42, BOX, "적용 조건 검토", "실험환경 vs 운영환경 차이"))
    s.append(arrow(cx, 338, cx, 364))
    # 분기: 근거 충분성 + 루프
    s.append(f'<polygon points="{cx},364 {cx+100},396 {cx},428 {cx-100},396" style="{DEC}"/>')
    s.append(f'<text x="{cx}" y="400" text-anchor="middle" style="{TXT}">근거 충분한가</text>')
    s.append(arrow(cx - 100, 396, 96, 396, "부족", dash=True))
    s.append(f'<line x1="96" y1="396" x2="96" y2="31" stroke="#55636f" stroke-width="1.2" '
             f'stroke-dasharray="4 3"/>')
    s.append(arrow(96, 31, cx - 110, 31))
    s.append(f'<text x="104" y="212" style="{SUB}">검색 예산 남으면 재검색</text>')
    s.append(arrow(cx, 428, cx, 454, "충분 또는 한도"))
    s.append(box(cx - 110, 454, 220, 42, ACC, "적합성 분석", "구조화 출력 (판정 + 주장)"))

    # 품질 검사는 순차 실행이다. 화살표 없이 나란히 두면 한 곳만 거치는 것처럼 읽힌다.
    row_y, row_h = 530, 42
    g_x, g_w = 14, 152
    l_x, l_w = 198, 140
    j_x, j_w = 370, 180
    s.append(f'<line x1="{cx}" y1="496" x2="{cx}" y2="512" stroke="#55636f" stroke-width="1.2"/>')
    s.append(f'<line x1="{cx}" y1="512" x2="{g_x + g_w / 2}" y2="512" stroke="#55636f" stroke-width="1.2"/>')
    s.append(arrow(g_x + g_w / 2, 512, g_x + g_w / 2, row_y))
    s.append(box(g_x, row_y, g_w, row_h, BOX, "guard", "결정적 검증 6종"))
    s.append(arrow(g_x + g_w, row_y + row_h / 2, l_x, row_y + row_h / 2))
    s.append(box(l_x, row_y, l_w, row_h, BOX, "linter", "표현 검사"))
    s.append(arrow(l_x + l_w, row_y + row_h / 2, j_x, row_y + row_h / 2))
    s.append(box(j_x, row_y, j_w, row_h, BOX, "judge", "coverage·neutrality"))
    exit_x = j_x + j_w / 2
    s.append(f'<line x1="{exit_x}" y1="{row_y + row_h}" x2="{exit_x}" y2="596" stroke="#55636f" stroke-width="1.2"/>')
    s.append(f'<line x1="{exit_x}" y1="596" x2="{cx}" y2="596" stroke="#55636f" stroke-width="1.2"/>')
    s.append(arrow(cx, 596, cx, 616))
    s.append(box(cx - 130, 616, 260, 44, OKB, "domain_findings 반환", "관점 전용 키만 갱신"))
    s.append("</svg>")
    return "".join(s)


# --- HTML 조립 --------------------------------------------------------------
CSS = """
@page { size: A4; margin: 17mm 16mm 18mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: "Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans KR", sans-serif;
  font-size: 10.4pt; line-height: 1.72; color: #1c2530; margin: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
.cover { border-bottom: 2.5px solid #2f5d8a; padding-bottom: 14px; margin-bottom: 26px; }
.cover .kicker { font-size: 9.5pt; letter-spacing: .06em; color: #2f5d8a; font-weight: 600; }
.cover h1 { font-size: 19pt; margin: 8px 0 6px; letter-spacing: -.02em; }
.cover .meta { font-size: 9.2pt; color: #5b6a78; }
h2 {
  font-size: 13pt; margin: 26px 0 10px; padding-bottom: 6px;
  border-bottom: 1.4px solid #d4dce4; break-after: avoid; page-break-after: avoid;
}
h3 { font-size: 11pt; margin: 18px 0 7px; color: #2f5d8a; break-after: avoid; }
p { margin: 9px 0; }
ol, ul { margin: 9px 0; padding-left: 21px; }
li { margin: 4px 0; }
table {
  width: 100%; border-collapse: collapse; margin: 13px 0; font-size: 9.3pt;
  break-inside: avoid; page-break-inside: avoid;
}
th, td { border: 1px solid #ccd5dd; padding: 6px 9px; text-align: left; vertical-align: top; }
th { background: #eef2f6; font-weight: 600; }
tbody tr:nth-child(even) { background: #fafbfc; }
code {
  font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 9pt;
  background: #f1f4f7; padding: 1px 4px; border-radius: 3px;
}
pre {
  background: #f7f9fb; border: 1px solid #dde4ea; border-left: 3px solid #2f5d8a;
  border-radius: 4px; padding: 10px 12px; margin: 12px 0;
  white-space: pre-wrap; word-break: break-word; overflow: visible;
  break-inside: avoid; page-break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: 8.8pt; line-height: 1.55; }
hr { border: none; border-top: 1px solid #d4dce4; margin: 22px 0; }
.figure {
  border: 1px solid #dde4ea; border-radius: 6px; padding: 14px; margin: 16px 0;
  background: #fcfdfe; break-inside: avoid; page-break-inside: avoid;
}
.figure .cap { font-size: 8.8pt; color: #5b6a78; margin-top: 8px; text-align: center; }
strong { font-weight: 600; }
@media print { table, pre, .figure { break-inside: avoid; } h2, h3 { break-after: avoid; } }
"""


def main() -> None:
    text = SRC.read_text(encoding="utf-8")

    # 제목·머리말은 표지로 옮긴다
    text = re.sub(r"^# .*?\n", "", text, count=1)
    text = text.replace(
        "팀 보고서에 들어갈 담당 파트입니다. 전체 보고서의 SUMMARY·REFERENCE 와 나머지 관점은\n"
        "다른 담당자 분량이며, 여기서는 도메인 관점 평가와 RAG 파이프라인만 다룹니다.\n\n---\n", ""
    )

    # mermaid 블록을 SVG 로 교체
    placeholder = "@@DIAGRAM@@"
    text = re.sub(r"```mermaid.*?```", placeholder, text, flags=re.S)

    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    figure = (f'<div class="figure">{build_svg()}'
              f'<div class="cap">도메인 평가 에이전트 서브그래프 — 분기 2곳, 루프 1곳</div></div>')
    body = body.replace(f"<p>{placeholder}</p>", figure).replace(placeholder, figure)

    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>도메인 평가 에이전트 및 RAG 적용</title><style>{CSS}</style></head><body>
<div class="cover">
  <div class="kicker">KV cache 최적화 기술 다관점 평가 · 담당 파트</div>
  <h1>도메인 평가 에이전트 및 RAG 적용</h1>
  <div class="meta">평가 도메인 : 데이터센터 &nbsp;·&nbsp; 대상 기술 : DeepSeek-V2 MLA (SW) / ITME CXL-Hybrid (HW)<br>
  기준일 2026-09-22 &nbsp;·&nbsp; 전체 보고서의 SUMMARY·REFERENCE 및 나머지 관점은 별도 담당</div>
</div>
{body}
</body></html>"""

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"HTML: {OUT_HTML} ({OUT_HTML.stat().st_size:,}B)")

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(OUT_HTML.as_uri())
        page.wait_for_load_state("networkidle")
        page.pdf(
            path=str(OUT_PDF),
            format="A4",
            print_background=True,  # 배경색·테두리가 날아가지 않게
            margin={"top": "17mm", "bottom": "18mm", "left": "16mm", "right": "16mm"},
            display_header_footer=True,
            header_template="<div></div>",
            footer_template=(
                '<div style="width:100%;font-size:8px;color:#8a97a3;'
                'padding:0 16mm;font-family:sans-serif;">'
                '<span style="float:left">도메인 평가 에이전트 및 RAG 적용</span>'
                '<span style="float:right"><span class="pageNumber"></span>'
                ' / <span class="totalPages"></span></span></div>'
            ),
        )
        browser.close()
    print(f"PDF : {OUT_PDF} ({OUT_PDF.stat().st_size:,}B)")
    subprocess.run(["pdfinfo", str(OUT_PDF)], check=False)


if __name__ == "__main__":
    main()
