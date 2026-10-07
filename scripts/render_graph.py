"""Supervisor 그래프를 PNG 로 그린다 (README Architecture 용).

  python -m scripts.render_graph            # outputs/agent-architecture/png/00-supervisor-graph.png

실제 `graph/build.py` 에서 edge 를 읽어 그리므로 코드와 그림이 어긋나지 않는다. matplotlib 만 필요하다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WORKERS = ("technical", "market", "stakeholder", "domain", "synthesis", "report", "quality")
LABELS = {"supervisor": "Supervisor\n(State 보고 다음 노드 결정)", "technical": "① technical", "market": "② market", "stakeholder": "③ stakeholder",
          "domain": "④ domain", "synthesis": "⑤ synthesis", "report": "⑥ report", "quality": "품질 평가\nquality"}


def edges() -> list[tuple[str, str, bool]]:
    from graph.build import build_graph

    noop = lambda state: {}
    app = build_graph(technical=noop, market=noop, stakeholder=noop, domain=noop, synthesis=noop, report=noop)
    return [(e.source, e.target, bool(e.conditional)) for e in app.get_graph().edges]


def draw(path: Path) -> None:
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    for font in ("/System/Library/Fonts/Supplemental/AppleGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if Path(font).exists():
            fm.fontManager.addfont(font)
            plt.rcParams["font.family"] = fm.FontProperties(fname=font).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False

    positions = {"supervisor": (0.0, 0.0)}
    for index, name in enumerate(WORKERS):
        angle = math.pi / 2 - 2 * math.pi * index / len(WORKERS)
        positions[name] = (4.2 * math.cos(angle), 3.1 * math.sin(angle))
    positions["__start__"], positions["__end__"] = (-5.9, -3.3), (5.9, -3.3)

    fig, ax = plt.subplots(figsize=(11, 7.6))
    ax.set_xlim(-6.8, 6.8)
    ax.set_ylim(-4.6, 4.4)
    ax.axis("off")

    boxes = {}
    for name, (x, y) in positions.items():
        if name.startswith("__"):
            boxes[name] = FancyBboxPatch((x - 0.55, y - 0.28), 1.1, 0.56, boxstyle="round,pad=0.04", linewidth=1.2,
                                         edgecolor="#666666", facecolor="#eeeeee")
            ax.add_patch(boxes[name])
            ax.text(x, y, "START" if name == "__start__" else "END", ha="center", va="center", fontsize=11)
            continue
        width, height = (3.4, 1.2) if name == "supervisor" else (2.1, 0.8)
        boxes[name] = FancyBboxPatch((x - width / 2, y - height / 2), width, height, boxstyle="round,pad=0.05", linewidth=1.5,
                                     edgecolor="#2b4f81", facecolor="#fbe9d0" if name == "supervisor" else "#eaf1fb")
        ax.add_patch(boxes[name])
        ax.text(x, y, LABELS[name], ha="center", va="center", fontsize=10.5, color="#12233f")

    for source, target, conditional in edges():
        # 왕복하는 두 edge 가 겹치지 않게 서로 반대 방향으로 휜다. 화살표는 상자 경계에서 시작하고 끝난다.
        ax.add_patch(FancyArrowPatch(
            positions[source], positions[target], patchA=boxes[source], patchB=boxes[target], arrowstyle="-|>", mutation_scale=14,
            connectionstyle=f"arc3,rad={-0.14 if conditional else -0.14}", shrinkA=2, shrinkB=2,
            color="#2b4f81" if conditional else "#8a1f1f", linewidth=1.4, linestyle="--" if conditional else "-"))
    ax.text(0, -4.45, "파랑 점선: supervisor 의 조건부 edge (State 로 결정)    빨강 실선: 워커 → supervisor 복귀 (직접 통신 없음)",
            ha="center", fontsize=9.5, color="#444444")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser(description="Supervisor 그래프 PNG")
    parser.add_argument("--output", type=Path, default=Path("outputs/agent-architecture/png/00-supervisor-graph.png"))
    args = parser.parse_args()
    draw(args.output)
    print(f"저장: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
