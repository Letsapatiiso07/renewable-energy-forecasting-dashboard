"""
Naive baselines for a day-ahead forecast issued at midnight for the next
24 hours. Both use only output observed before the forecast is issued.
"""
import pandas as pd


def _lookup(series: pd.Series, times: pd.Series) -> pd.Series:
    return pd.Series(series.reindex(times).to_numpy(), index=times.index)


def persistence(actual: pd.DataFrame, column: str) -> pd.Series:
    """Last value before midnight (23:00 the day before), held for the whole day"""
    s = actual.set_index("time")[column]
    issue = actual["time"].dt.normalize() - pd.Timedelta(hours=1)
    return _lookup(s, issue)


def seasonal_naive(actual: pd.DataFrame, column: str) -> pd.Series:
    """Same hour on the previous day"""
    s = actual.set_index("time")[column]
    return _lookup(s, actual["time"] - pd.Timedelta(hours=24))
