"""Matplotlib rendering for scripts/analytics.py: courts, heatmaps and charts as PNG images.

Colors are the documented light palette (blue/orange categorical slots validated for CVD and contrast,
a one-hue blue ramp for density, blue<->red around a gray midpoint for win rates). Sizes are given
in points so marks keep their spec at any export resolution: bars at most 18 pt thick with 3 pt
rounded tops, 1.5 pt lines, markers 6 pt across with a surface ring, hairline solid gridlines.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, DEEMPH = "#2a78d6", "#eb6834", "#898781"
ORD1, ORD2 = "#86b6ef", "#184f95"  # ordinal pair: 1st serve -> 2nd serve
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281", "#0d366b"]
DIV_MID, DIV_POS, DIV_NEG = "#f0efec", "#2a78d6", "#e34948"
BL, SVC, SGL, DBL, POST = 11.885, 6.40, 4.115, 5.485, 6.40
DPI = 200
HALF_VIEW = dict(x=(-7.5, 7.5), y=(-19.5, 0.6))

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9,
    "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.linewidth": 0.75,
})


def _hex(c: str) -> np.ndarray:
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)], float) / 255


_SEQ = np.array([_hex(c) for c in SEQ])


def seq_rgba(v: np.ndarray) -> np.ndarray:
    """Density in [0, 1] -> RGBA on the blue ramp; the lowest values fade out so court lines show."""
    v = np.clip(v, 0, 1)
    t = v * (len(_SEQ) - 1)
    i = np.minimum(np.floor(t).astype(int), len(_SEQ) - 2)
    f = (t - i)[..., None]
    rgb = _SEQ[i] * (1 - f) + _SEQ[i + 1] * f
    a = np.clip((v - 0.03) / 0.10, 0, 1)[..., None]
    return np.concatenate([rgb, a], axis=-1)


def div_color(t: float) -> np.ndarray:
    """t in [-1, 1]: negative -> red, positive -> blue, 0 -> gray."""
    u = float(np.clip(t, -1, 1))
    return _hex(DIV_MID) * (1 - abs(u)) + _hex(DIV_POS if u >= 0 else DIV_NEG) * abs(u)


# ---------------------------------------------------------------- figure scaffolding

def figure(w: float, h: float, title: str, subtitle: str = "", note: str = ""):
    """Fixed-layout figure (inches) with a title block; returns (fig, top) where `top` is the figure
    fraction below the title block."""
    fig = plt.figure(figsize=(w, h), dpi=DPI)
    y = 1 - 0.32 / h
    fig.text(0.3 / w, y, title, fontsize=13, fontweight="bold", color=INK, va="top")
    lines = _wrap(subtitle, int(w * 15.5)) if subtitle else []
    for k, line in enumerate(lines):
        fig.text(0.3 / w, y - (0.3 + 0.19 * k) / h, line, fontsize=9.5, color=INK2, va="top")
    if note:
        for k, line in enumerate(_wrap(note, int(w * 19))[::-1]):
            fig.text(0.3 / w, (0.16 + 0.15 * k) / h, line, fontsize=7.5, color=MUTED, va="bottom")
    top = y - (0.42 + 0.19 * len(lines)) / h
    return fig, top


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width) or [""]


def save(fig, path):
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def legend(fig, x: float, y: float, items: list[tuple[str, str, str]]):
    """items: (kind, color, label) with kind 'box' | 'line' | 'dot'; placed in figure fractions."""
    w, _ = fig.get_size_inches()
    for kind, color, label in items:
        if kind == "box":
            fig.patches.append(Rectangle((x, y - 0.012), 0.11 / w, 0.022, transform=fig.transFigure, color=color, lw=0))
        elif kind == "line":
            fig.lines.append(plt.Line2D([x, x + 0.16 / w], [y, y], transform=fig.transFigure, color=color, lw=1.5,
                                        solid_capstyle="round"))
        else:
            fig.lines.append(plt.Line2D([x + 0.055 / w], [y], transform=fig.transFigure, marker="o", ms=6, color=color,
                                        mec=SURFACE, mew=1.5, lw=0))
        t = fig.text(x + 0.22 / w, y, label, fontsize=8.5, color=INK2, va="center")
        x += (0.22 + 0.075 * len(label) + 0.2) / w
    return t


def value_axis(ax, ticks, fmt, label: str | None = None):
    ax.set_yticks(ticks)
    ax.set_yticklabels([fmt(t) for t in ticks])
    ax.set_ylim(ticks[0], ticks[-1])
    ax.grid(axis="y", color=GRID, lw=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", pad=4, labelsize=8)
    ax.tick_params(axis="x", labelsize=8, pad=4)
    if label:
        ax.set_xlabel(label, fontsize=8, color=MUTED, labelpad=6)


def _pt_to_data(ax, pts: float) -> tuple[float, float]:
    bb = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    inch = pts / 72 * ax.figure.dpi
    return inch * (x1 - x0) / bb.width, inch * (y1 - y0) / bb.height


def columns(ax, cats: list[str], series: list[tuple[str, str]], values: dict[str, list], labels=None, ref=None):
    """Grouped columns: series = [(name, color)], values[name][i]. At most 18 pt thick, 1.5 pt gap,
    3 pt rounded top, square at the baseline. `labels`: formatter for value labels on the caps."""
    ax.set_xlim(-0.5, len(cats) - 0.5)
    rx, ry = _pt_to_data(ax, 3)
    gap, _ = _pt_to_data(ax, 1.5)
    bw = min(_pt_to_data(ax, 18)[0], (0.72 - gap * (len(series) - 1)) / len(series))
    gw = bw * len(series) + gap * (len(series) - 1)
    for i in range(len(cats)):
        for k, (name, color) in enumerate(series):
            v = values[name][i]
            if v is None or not np.isfinite(v):
                continue
            x = i - gw / 2 + k * (bw + gap)
            r = min(rx, bw / 2)
            ax.add_patch(FancyBboxPatch((x, -ry), bw, v + ry, boxstyle=f"round,pad=0,rounding_size={r}",
                                        mutation_aspect=ry / rx, fc=color, ec="none", zorder=3))
            if labels:
                ax.text(x + bw / 2, v, labels(v), ha="center", va="bottom", fontsize=7.5, color=INK2, zorder=4,
                        transform=ax.transData).set_y(v + ry * 1.2)
    if ref is not None:
        ax.axhline(ref, color=AXIS, lw=0.75, zorder=2)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats)
    ax.spines["bottom"].set_color(AXIS)


# ---------------------------------------------------------------- courts

def court(ax, view=HALF_VIEW, full=False, surface=True, lw=0.8):
    ends = (-1, 1) if full else (-1,)
    segs = []
    for e in ends:
        segs += [[(-DBL, e * BL), (DBL, e * BL)], [(-DBL, 0), (-DBL, e * BL)], [(DBL, 0), (DBL, e * BL)],
                 [(-SGL, 0), (-SGL, e * BL)], [(SGL, 0), (SGL, e * BL)], [(-SGL, e * SVC), (SGL, e * SVC)],
                 [(0, 0), (0, e * SVC)], [(0, e * BL), (0, e * (BL - 0.3))]]
    halo = [pe.Stroke(linewidth=lw + 2.2, foreground=SURFACE, alpha=0.8), pe.Normal()]
    ax.add_collection(LineCollection(segs, colors=INK2, linewidths=lw, zorder=4, capstyle="round", path_effects=halo))
    ax.plot([-POST, POST], [0, 0], color=INK2, lw=lw * 2, zorder=4, solid_capstyle="round", path_effects=[
        pe.Stroke(linewidth=lw * 2 + 2.2, foreground=SURFACE, alpha=0.8), pe.Normal()])
    if surface:
        ax.add_patch(Rectangle((-DBL, -BL), 2 * DBL, BL * (2 if full else 1), color=GRID, alpha=0.45, lw=0, zorder=0))
    ax.set_xlim(*view["x"])
    ax.set_ylim(*view["y"])
    ax.set_aspect("equal")
    ax.axis("off")


def heat(ax, panel: dict):
    """panel: dens (rows from the top edge, values scaled to [0, 1]), x0, x1, y0, y1."""
    ax.imshow(seq_rgba(panel["dens"]), extent=(panel["x0"], panel["x1"], panel["y0"], panel["y1"]), origin="upper",
              interpolation="bilinear", zorder=2)


def heat_scale(fig, x: float, y: float, w: float, peak: float, unit: str):
    """Horizontal scale bar in figure fractions for the shared density scale."""
    cax = fig.add_axes([x, y, w, 0.012])
    v = np.linspace(0, 1, 256)[None, :]
    rgba = seq_rgba(v)
    bg = _hex(SURFACE)
    cax.imshow(rgba[..., :3] * rgba[..., 3:] + bg * (1 - rgba[..., 3:]), aspect="auto", extent=(0, 1, 0, 1))
    cax.axis("off")
    fig.text(x - 0.008, y + 0.006, "0", ha="right", va="center", fontsize=7.5, color=MUTED)
    fig.text(x + w + 0.008, y + 0.006, f"{peak:.1f}% of {unit} per m²", ha="left", va="center", fontsize=7.5, color=MUTED)


def court_row(fig, top: float, bottom: float, panels: list[tuple[dict, str, str]], view=HALF_VIEW, full=False,
              gap_in: float = 0.5, left_in: float = 0.3):
    """Side-by-side courts between figure fractions top/bottom; returns the axes."""
    w, h = fig.get_size_inches()
    ph = (top - bottom) * h - 0.42
    aspect = (view["x"][1] - view["x"][0]) / (view["y"][1] - view["y"][0])
    pw = ph * aspect
    axes = []
    for k, (panel, title, sub) in enumerate(panels):
        x = (left_in + k * (pw + gap_in)) / w
        fig.text(x, top, title, fontsize=10, fontweight="bold", va="top")
        fig.text(x, top - 0.2 / h, sub, fontsize=8, color=MUTED, va="top")
        ax = fig.add_axes([x, bottom, pw / w, ph / h])
        court(ax, view, full)
        if panel is not None:
            heat(ax, panel)
        axes.append(ax)
    return axes, pw


def mini_grid(path, items: list[dict], title: str, subtitle: str, note: str, label, sub, cols: int = 8,
              groups: list[tuple[str, list[dict]]] | None = None):
    """Small-multiple courts (each scaled to its own peak). items: dicts with 'panel'."""
    groups = groups or [("", items)]
    view = dict(x=(-7, 7), y=(-18.5, 0.6))
    cw = 1.25
    ch = cw * (view["y"][1] - view["y"][0]) / (view["x"][1] - view["x"][0])
    rows = sum(int(np.ceil(len(g) / cols)) for _, g in groups)
    W = 0.6 + cols * (cw + 0.18)
    H = 1.3 + rows * (ch + 0.55) + 0.45 * len([g for g, _ in groups if g]) + 0.5
    fig, top = figure(W, H, title, subtitle, note)
    y = top * H
    for gname, gitems in groups:
        if gname:
            fig.text(0.3 / W, y / H, gname, fontsize=10, fontweight="bold", color=INK2, va="top")
            y -= 0.35
        for k, it in enumerate(gitems):
            r, c = divmod(k, cols)
            x0 = 0.3 + c * (cw + 0.18)
            yt = y - r * (ch + 0.55)
            fig.text(x0 / W, yt / H, label(it), fontsize=7.5, fontweight="bold", va="top")
            ax = fig.add_axes([x0 / W, (yt - 0.17 - ch) / H, cw / W, ch / H])
            court(ax, view, lw=0.5)
            heat(ax, it["panel"])
            fig.text(x0 / W, (yt - 0.2 - ch) / H, sub(it), fontsize=6.5, color=MUTED, va="top")
        y -= int(np.ceil(len(gitems) / cols)) * (ch + 0.55) + 0.1
    save(fig, path)
