"""
Error metrics and skill scores on the held-out test period
"""
from typing import Dict

import numpy as np
import pandas as pd

METHODS = ["model", "persistence", "seasonal_naive", "physical_nwp"]
BASELINES = ["persistence", "seasonal_naive", "physical_nwp"]


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def score_frame(df: pd.DataFrame) -> Dict:
    """`df` holds `actual` and one column per method"""
    scores = {m: {"mae": mae(df["actual"], df[m]), "rmse": rmse(df["actual"], df[m])}
              for m in METHODS}
    # Skill = 1 - model error / baseline error; above 0 means the model wins
    scores["skill_vs"] = {
        b: {k: (1 - scores["model"][k] / scores[b][k]) if scores[b][k] > 0 else None
            for k in ("mae", "rmse")}
        for b in BASELINES
    }
    scores["n_hours"] = int(len(df))
    scores["mean_actual_mw"] = float(df["actual"].mean())
    return scores


def evaluate(df: pd.DataFrame) -> Dict:
    """Scores overall and per site"""
    return {
        "overall": score_frame(df),
        "by_site": {site: score_frame(g) for site, g in df.groupby("location")},
    }
