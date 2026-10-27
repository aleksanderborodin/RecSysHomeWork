from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

BASE, ACCENT = "tab:blue", "tab:orange"


def use() -> None:
    plt.style.use("default")
    mpl.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100, "savefig.dpi": 150})


def figure(w: float = 12, h: float = 4, ncols: int = 1, nrows: int = 1):
    return plt.subplots(nrows, ncols, figsize=(w, h), layout="constrained")


def fmt_int(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ")


def thousands(ax, axis="y") -> None:
    f = mpl.ticker.FuncFormatter(lambda v, _: fmt_int(v))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def barh(ax, names, values, lo=None, hi=None, highlight=None, fmt="{:.3f}", xlabel=None):

    order = np.argsort(values)
    names, values = np.asarray(names)[order], np.asarray(values, dtype=float)[order]
    colors = [ACCENT if n == highlight else BASE for n in names]
    xerr = None
    if lo is not None:
        lo, hi = np.asarray(lo)[order], np.asarray(hi)[order]
        xerr = [values - lo, hi - values]
    bars = ax.barh(names, values, color=colors, xerr=xerr, capsize=3)
    ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=4 if xerr is None else 18)
    ax.set_xlim(0, (values.max() if hi is None else hi.max()) * 1.2)
    ax.grid(axis="y", visible=False)
    if xlabel:
        ax.set_xlabel(xlabel)


def sweep(ax, x, lines: dict, best=None, xlog=False, xlabel=None, ylabel=None, legend_title=None):

    for label, ys in lines.items():
        ax.plot(x, ys, marker="o", ms=4, label=label)
    if best is not None:
        ax.plot(*best, "*", ms=16, color="red", zorder=5, label="выбрано")
    if xlog:
        ax.set_xscale("log")
        ax.set_xticks(x, [f"{v:g}" for v in x])
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(title=legend_title, fontsize=9)


def ablation(ax, names, deltas, lo, hi, xlabel="Δ NDCG@10 (95% CI)"):

    y = np.arange(len(names))[::-1]
    deltas = np.asarray(deltas, dtype=float)
    ax.errorbar(deltas, y, xerr=[deltas - np.asarray(lo), np.asarray(hi) - deltas], fmt="o", capsize=4)
    for yi, d, h in zip(y, deltas, hi):
        ax.annotate(f"{d:+.4f}", (h, yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y, names)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.margins(x=0.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)


def heat(ax, mat, rows, cols, fmt="{:.3f}", cmap="Blues", cbar_label=None):
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    hi = np.nanmax(mat)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            ax.text(c, r, fmt.format(mat[r, c]), ha="center", va="center", fontsize=9,
                    color="white" if mat[r, c] > 0.6 * hi else "black")
    ax.figure.colorbar(im, ax=ax, label=cbar_label)
