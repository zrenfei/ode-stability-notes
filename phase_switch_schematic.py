from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


COLORS = {
    "blue": "#365CAD",
    "orange": "#D96C2C",
    "green": "#4C8A47",
    "gray": "#5A5A5A",
    "teal": "#2A9D8F",
    "ink": "#222222",
    "grid": "#D7D7D7",
    "panel_bg": "#FBFAF7",
    "label_bg": "#FFFDF8",
}

NOTE_BOX = dict(
    boxstyle="round,pad=0.28",
    facecolor=COLORS["label_bg"],
    edgecolor="none",
    alpha=0.92,
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 260,
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.titlepad": 7,
            "axes.linewidth": 0.9,
            "axes.facecolor": COLORS["panel_bg"],
            "axes.edgecolor": "#444444",
            "grid.color": COLORS["grid"],
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.45,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "stix",
            "text.usetex": False,
        }
    )


def style_axes(ax: Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.tick_params(direction="out", length=4, width=0.85)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def add_panel_title(ax: Axes, tag: str, title: str) -> None:
    ax.set_title(f"({tag}) {title}", loc="left", fontweight="bold")


def make_figure() -> Figure:
    configure_style()
    fig = plt.figure(figsize=(12.4, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[0.12, 1.0, 1.0, 0.18])
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    axs = np.array(
        [
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
            [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
        ]
    )
    caption_ax = fig.add_subplot(gs[3, :])
    caption_ax.axis("off")
    fig.patch.set_facecolor("white")
    title_ax.text(
        0.5,
        0.15,
        "Phase-aware geometry and switching time",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color=COLORS["ink"],
    )

    # (a) Potential and smoothing
    ax = axs[0, 0]
    x = np.linspace(-2.2, 2.2, 500)
    v0 = 0.42 * (x**2 - 1.0) ** 2 + 0.035 * x + 0.08
    vs = 0.18 * (x**2 - 0.55) ** 2 + 0.08 * x**2 + 0.17
    ax.plot(x, v0, color=COLORS["gray"], linewidth=2.8)
    ax.plot(x, vs, color=COLORS["blue"], linewidth=2.8, linestyle=(0, (5, 3)))
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-0.05, 0.92)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("potential")
    add_panel_title(ax, "a", "Data and smoothing")
    ax.text(-1.70, 0.76, r"$V_0$ double-well", color=COLORS["gray"], fontsize=8.7, bbox=NOTE_BOX)
    ax.text(0.90, 0.16, r"$V_s$ smoothed", color=COLORS["blue"], fontsize=8.7, bbox=NOTE_BOX)
    ax.annotate(
        "Local nonconvexity\ncan remain",
        xy=(0.02, 0.27),
        xytext=(-0.35, 0.40),
        fontsize=8.5,
        color=COLORS["blue"],
        bbox=NOTE_BOX,
        arrowprops=dict(arrowstyle="->", color=COLORS["blue"], lw=1.5),
    )
    ax.annotate(
        "Large-scale geometry\nbecomes smoother",
        xy=(1.08, 0.43),
        xytext=(0.85, 0.69),
        fontsize=8.5,
        color=COLORS["gray"],
        bbox=NOTE_BOX,
        arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.5),
    )
    style_axes(ax)

    # (b) Two-scale profile
    ax = axs[0, 1]
    r = np.linspace(0.02, 2.35, 400)
    kappa = 0.43 * np.tanh(2.8 * (r - 1.08)) - 0.03
    switch_radius = 1.05
    lower_bound = np.where(r <= switch_radius, -0.45, 0.35)
    ax.axvspan(0.0, switch_radius, color=COLORS["orange"], alpha=0.07, lw=0)
    ax.axvspan(switch_radius, 2.45, color=COLORS["blue"], alpha=0.05, lw=0)
    ax.axhline(0.0, color=COLORS["ink"], linewidth=1.0, alpha=0.4)
    ax.plot(r, kappa, color=COLORS["blue"], linewidth=2.8)
    ax.plot(
        r,
        lower_bound,
        color=COLORS["orange"],
        linewidth=2.6,
        linestyle=(0, (6, 3)),
    )
    ax.axvline(switch_radius, color=COLORS["ink"], linestyle=":", linewidth=1.2, alpha=0.65)
    ax.text(0.16, -0.11, "short-range instability", color=COLORS["orange"], fontsize=8.5, bbox=NOTE_BOX)
    ax.text(1.30, 0.18, "large-scale contraction", color=COLORS["blue"], fontsize=8.5, bbox=NOTE_BOX)
    ax.text(1.55, 0.38, r"$\kappa_{\widehat{b}_t}(r)$", color=COLORS["blue"], fontsize=8.7)
    ax.text(1.48, 0.29, r"$\kappa^{\mathrm{lb}}_{s_0}(r)$", color=COLORS["orange"], fontsize=8.7)
    ax.text(switch_radius + 0.02, -0.58, r"$R(s_0)$", fontsize=8.7, color=COLORS["ink"])
    ax.set_xlim(0, 2.45)
    ax.set_ylim(-0.62, 0.52)
    ax.set_xticks([0, switch_radius, 2.3])
    ax.set_xticklabels([r"$0$", "", ""])
    ax.set_yticks([-0.45, 0, 0.35])
    ax.set_yticklabels([r"$-L_{\mathfrak{b}}(s_0)$", r"$0$", r"$m(s_0)$"])
    ax.set_xlabel(r"separation $r$")
    ax.set_ylabel("drift profile")
    add_panel_title(ax, "b", "Early two-scale geometry")
    style_axes(ax)

    # (c) Margin/load and admissibility
    ax = axs[1, 0]
    s = np.linspace(0, 1, 400)
    margin = -0.22 + 0.82 * s**1.6
    load = 0.56 * np.exp(-1.8 * s) + 0.08
    s_min = 0.46
    ax.axvspan(s_min, 1.02, color=COLORS["green"], alpha=0.10, lw=0)
    ax.axhline(0.0, color=COLORS["ink"], linewidth=1.0, alpha=0.4)
    ax.plot(s, margin, color=COLORS["blue"], linewidth=2.8)
    ax.plot(s, load, color=COLORS["orange"], linewidth=2.8)
    ax.axvline(s_min, color=COLORS["ink"], linestyle=":", linewidth=1.2, alpha=0.65)
    ax.text(0.68, 0.61, "admissible window", color=COLORS["green"], fontsize=8.5, bbox=NOTE_BOX)
    ax.text(0.67, 0.26, r"$\mathfrak{m}(s)$", color=COLORS["blue"], fontsize=9)
    ax.text(0.16, 0.44, r"$\mathfrak{b}(s)$", color=COLORS["orange"], fontsize=9)
    ax.text(
        0.06,
        -0.12,
        "Too early:\nno uniform margin",
        color=COLORS["gray"],
        fontsize=8.5,
        bbox=NOTE_BOX,
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.34, 0.74)
    ax.set_xticks([0, s_min, 1.0])
    ax.set_xticklabels([r"$0$", r"$s_{\min}$", r"$T$"])
    ax.set_yticks([0])
    ax.set_yticklabels([r"$0$"])
    ax.set_xlabel(r"forward time $s$")
    ax.set_ylabel("size")
    add_panel_title(ax, "c", "Margin and admissibility")
    style_axes(ax)

    # (d) Switch balance
    ax = axs[1, 1]
    s0 = np.linspace(0.46, 1.0, 400)
    early = 0.24 + 0.13 * np.exp(-6.0 * (s0 - 0.46))
    late = 0.08 + 0.46 * (s0 - 0.46) ** 1.3
    conversion = 0.10 + 0.03 / (s0 - 0.445)
    total = early + late + conversion - 0.05
    best_index = int(np.argmin(total))
    s0_star = float(s0[best_index])
    total_star = float(total[best_index])
    ax.plot(s0, early, color=COLORS["blue"], linewidth=2.5)
    ax.plot(s0, conversion, color=COLORS["teal"], linewidth=2.5)
    ax.plot(s0, late, color=COLORS["orange"], linewidth=2.5)
    ax.plot(s0, total, color=COLORS["ink"], linewidth=3.0)
    ax.axvline(s0_star, color=COLORS["ink"], linestyle=":", linewidth=1.2, alpha=0.65)
    ax.scatter([s0_star], [total_star], color=COLORS["ink"], s=28, zorder=5)
    ax.annotate(
        r"optimal switch $s_0^\star$",
        xy=(s0_star, total_star),
        xytext=(0.69, 0.55),
        fontsize=8.5,
        color=COLORS["ink"],
        bbox=NOTE_BOX,
        arrowprops=dict(arrowstyle="->", color=COLORS["ink"], lw=1.4),
    )
    ax.text(0.48, 0.34, "early damping", color=COLORS["blue"], fontsize=8.7)
    ax.text(0.49, 0.57, "conversion cost", color=COLORS["teal"], fontsize=8.7)
    ax.text(0.82, 0.15, "late amplification", color=COLORS["orange"], fontsize=8.7)
    ax.text(0.74, 0.45, r"$\mathcal{B}_p(s_0)$", color=COLORS["ink"], fontsize=9)
    ax.text(
        0.48,
        0.25,
        "Small $s_0$:\nmore damping,\nmore conversion cost",
        color=COLORS["teal"],
        fontsize=8.4,
        bbox=NOTE_BOX,
    )
    ax.text(
        0.81,
        0.19,
        "Large $s_0$:\nless damping,\nlonger late stage",
        color=COLORS["gray"],
        fontsize=8.4,
        bbox=NOTE_BOX,
    )
    ax.set_xlim(0.46, 1.0)
    ax.set_ylim(0.12, 0.62)
    ax.set_xticks([0.46, s0_star, 1.0])
    ax.set_xticklabels([r"$s_{\min}$", r"$s_0^\star$", r"$T$"])
    ax.set_yticks([])
    ax.set_xlabel(r"candidate switch $s_0$")
    ax.set_ylabel("contribution")
    add_panel_title(ax, "d", "Why the switch balances")
    style_axes(ax)

    caption = (
        "Schematic summary of the two-stage picture. Early reverse time retains short-range instability "
        "but gains large-scale contraction after smoothing; the admissible switch window appears only once "
        "that global margin is available. The preferred interior switch balances three effects: early damping, "
        "conversion cost, and late Euclidean amplification."
    )
    caption_ax.text(
        0.5,
        0.55,
        textwrap.fill(caption, 140),
        ha="center",
        va="center",
        fontsize=8.8,
        color="#444444",
    )
    return fig


def main() -> None:
    fig = make_figure()
    output_prefix = Path(__file__).with_name("phase_switch_schematic")
    fig.savefig(output_prefix.with_suffix(".png"), pad_inches=0.12)
    fig.savefig(output_prefix.with_suffix(".pdf"), pad_inches=0.12)
    plt.close(fig)
    print(f"Saved {output_prefix.with_suffix('.png')}")
    print(f"Saved {output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
