"""
Figures for the README, drawn from the held-out test forecasts
"""
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils.config import LOCATIONS

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
COLORS = {
    "actual": INK,
    "model": "#2a78d6",
    "physical_nwp": "#eb6834",
    "seasonal_naive": "#1baf7a",
    "persistence": "#eda100",
}
LABELS = {
    "actual": "Modelled actual",
    "model": "XGBoost forecast",
    "seasonal_naive": "Seasonal naive (same hour yesterday)",
    "physical_nwp": "Physical model on forecast weather",
    "persistence": "Persistence (last value before midnight)",
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def backtest_plot(forecasts: pd.DataFrame, site: str, start: str, days: int, path: Path):
    """Actual vs forecasts for one site over a window of the test year"""
    end = pd.Timestamp(start) + pd.Timedelta(days=days)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, facecolor=SURFACE)
    for ax, target in zip(axes, ("solar", "wind")):
        d = forecasts[(forecasts["target"] == target) & (forecasts["location"] == site)
                      & (forecasts["time"] >= start) & (forecasts["time"] < end)]
        for col, width in (("actual", 2.2), ("seasonal_naive", 1.4), ("model", 1.8)):
            ax.plot(d["time"], d[col], color=COLORS[col], linewidth=width, label=LABELS[col])
        ax.set_ylabel(f"{target.capitalize()} output (MW)", color=INK_2, fontsize=9)
        ax.set_ylim(0, 105)
        _style(ax)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8.5, frameon=False, ncol=3)
    fig.suptitle(
        f"Day-ahead forecasts on held-out data, {LOCATIONS[site]['name']} "
        f"(100 MW plants, output modelled from ERA5 weather)",
        color=INK, fontsize=11, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def mae_bar_chart(metrics: Dict, path: Path):
    """Test-year MAE per site for the model and each baseline"""
    methods = ["model", "physical_nwp", "seasonal_naive", "persistence"]
    sites = list(LOCATIONS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    width = 0.2
    for ax, target in zip(axes, ("solar", "wind")):
        by_site = metrics["targets"][target]["by_site"]
        for i, m in enumerate(methods):
            xs = [j + (i - 1.5) * width for j in range(len(sites))]
            ys = [by_site[s][m]["mae"] for s in sites]
            ax.bar(xs, ys, width=width * 0.9, color=COLORS[m], label=LABELS[m])
            if m == "model":
                for x, y in zip(xs, ys):
                    ax.text(x, y + 0.5, f"{y:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(len(sites)))
        ax.set_xticklabels([LOCATIONS[s]["name"].split(",")[0] for s in sites])
        hours = metrics["targets"][target]["hours_scored"]
        ax.set_title(f"{target.capitalize()} MAE, MW ({hours})", color=INK, fontsize=10, loc="left")
        _style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle("Mean absolute error on the held-out test year (lower is better)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
