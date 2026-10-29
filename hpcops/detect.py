"""Anomaly detectors scored against the injected faults, all calibrated to the same false-page budget.

Methods (each yields a node-minute score):
  static     fixed thresholds on raw metrics (temp 85 C, 5 ECC/min, 200 retransmits/min, 15% I/O wait),
             scaled together by one multiplier
  rolling_z  z-score of each metric against the node's own trailing 6 hours
  iforest    Isolation Forest on raw node-minute metrics
  residual   load-normalised residuals (features.Baseline): temperature given power, retransmits and
             I/O wait given utilisation, ECC count against its background rate

An alert needs 3 consecutive minutes above threshold. Alerts on one node are merged into episodes
(gaps <= 15 min). Episodes that match no injected fault are false; false episodes starting within
10 minutes of each other count as one page (one page per heavy job, not one per node). Each method's
threshold is set on the training weeks to the lowest value giving <= `budget` false pages per day, and
then frozen for the test weeks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .features import Baseline, ECC, IOW, P, RET, TEMP, UTIL, rolling_sum
from .sim import MINUTES_PER_DAY, World

METHODS = ["static", "rolling_z", "iforest", "residual"]


def score_static(w: World, base: Baseline, **_) -> np.ndarray:
    X = w.X
    s = np.nanmax(np.stack([X[..., TEMP] / 85, X[..., ECC] / 5, X[..., RET] / 200, X[..., IOW] / 15]), axis=0)
    return np.nan_to_num(s)


def score_rolling_z(w: World, base: Baseline, window: int = 360, **_) -> np.ndarray:
    X = w.X
    series = [X[..., TEMP], rolling_sum(X[..., ECC], 30), rolling_sum(X[..., RET], 5) / 5, rolling_sum(X[..., IOW], 5) / 5]
    zs = []
    for s in series:
        df = pd.DataFrame(s.T)
        mu = df.rolling(window, min_periods=60).mean().shift(1)
        sd = df.rolling(window, min_periods=60).std().shift(1)
        zs.append(((df - mu) / (sd + 1e-3)).values.T)
    z = np.nanmax(np.stack(zs), axis=0)
    z[~np.isfinite(X[..., TEMP])] = 0
    return np.nan_to_num(z)


def _if_features(w: World) -> np.ndarray:
    X = w.X
    f = np.stack([X[..., P], X[..., TEMP], rolling_sum(X[..., ECC], 30), rolling_sum(X[..., RET], 5) / 5,
                  rolling_sum(X[..., IOW], 5) / 5, X[..., UTIL]], -1)
    return f.reshape(-1, f.shape[-1])


def fit_iforest(train: list[World], seed: int = 0, n: int = 200_000) -> IsolationForest:
    F = np.concatenate([_if_features(w) for w in train])
    F = F[np.isfinite(F).all(1)]
    rng = np.random.default_rng(seed)
    F = F[rng.choice(len(F), min(n, len(F)), replace=False)]
    return IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1).fit(F)


def score_iforest(w: World, base: Baseline, model: IsolationForest = None, **_) -> np.ndarray:
    F = _if_features(w)
    ok = np.isfinite(F).all(1)
    s = np.zeros(len(F))
    s[ok] = -model.score_samples(F[ok])
    return s.reshape(w.X.shape[0], w.T)


def score_residual(w: World, base: Baseline, **_) -> np.ndarray:
    sig = base.signals(w)
    z = np.nanmax(np.stack([sig["z_temp"], sig["z_ret"], sig["z_iow"], sig["z_ecc"]]), axis=0)
    return np.nan_to_num(z)


SCORERS = {"static": score_static, "rolling_z": score_rolling_z, "iforest": score_iforest, "residual": score_residual}


def persistent(alert: np.ndarray, k: int = 3) -> np.ndarray:
    out = alert.copy()
    for j in range(1, k):
        out[:, j:] &= alert[:, :-j]
        out[:, :j] = False
    return out


def episodes(alert: np.ndarray, gap: int = 15) -> list[tuple[int, int, int]]:
    eps = []
    for n in range(alert.shape[0]):
        idx = np.flatnonzero(alert[n])
        if len(idx) == 0:
            continue
        breaks = np.flatnonzero(np.diff(idx) > gap)
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks], idx[-1]]
        eps.extend((n, int(a), int(b)) for a, b in zip(starts, ends))
    return eps


def match(eps, incidents, slack: int = 30):
    """Return (per-incident first detection minute or None, list of unmatched episodes)."""
    first = {i.id: None for i in incidents}
    unmatched = []
    for n, a, b in eps:
        hit = False
        for inc in incidents:
            impact = inc.crash if inc.crash is not None else inc.end
            if n in inc.nodes and a <= impact + slack and b >= inc.start:
                hit = True
                if a < impact and (first[inc.id] is None or a < first[inc.id]):
                    first[inc.id] = max(a, inc.start)
        if not hit:
            unmatched.append((n, a, b))
    return first, unmatched


def pages(unmatched, window: int = 10) -> int:
    starts = sorted(a for _, a, _ in unmatched)
    count, anchor = 0, None
    for s in starts:
        if anchor is None or s - anchor > window:
            count += 1
            anchor = s
    return count


def evaluate_threshold(scores: list[np.ndarray], worlds: list[World], thr: float):
    rows, false_pages, days = [], 0, 0.0
    for s, w in zip(scores, worlds):
        eps = episodes(persistent(s > thr))
        first, unmatched = match(eps, w.incidents)
        false_pages += pages(unmatched)
        days += w.T / MINUTES_PER_DAY
        for inc in w.incidents:
            d = first[inc.id]
            rows.append({"world": w.seed, "id": inc.id, "kind": inc.kind, "component": inc.component,
                         "detected": d is not None, "delay": None if d is None else d - inc.start,
                         "crash": inc.crash, "lead": None if (d is None or inc.crash is None) else inc.crash - d})
    return pd.DataFrame(rows), false_pages / days


def calibrate(scores: list[np.ndarray], worlds: list[World], budget: float = 1.0) -> float:
    flat = np.concatenate([s[s > 0].ravel() for s in scores])
    rng = np.random.default_rng(0)
    flat = flat[rng.choice(len(flat), min(len(flat), 2_000_000), replace=False)]
    lo, hi = np.quantile(flat, 0.9), flat.max()
    grid = np.unique(np.quantile(flat, 1 - np.geomspace(1e-1, 1e-6, 60)))
    grid = grid[(grid >= lo) & (grid <= hi)]
    for thr in grid:  # ascending: first threshold within budget
        _, fp = evaluate_threshold(scores, worlds, thr)
        if fp <= budget:
            return float(thr)
    return float(hi)


def summarise(df: pd.DataFrame, fp_day: float) -> dict:
    out = {"false_pages_per_day": fp_day, "recall": float(df.detected.mean()), "by_kind": {}}
    for k, g in df.groupby("kind"):
        det = g[g.detected]
        out["by_kind"][k] = {"incidents": int(len(g)), "recall": float(g.detected.mean()),
                             "median_delay_min": None if det.empty else float(det.delay.median())}
    crash = df[df.crash.notna()]
    warned = crash[crash.detected]
    out["crashes"] = int(len(crash))
    out["crashes_warned"] = float(crash.detected.mean()) if len(crash) else None
    out["median_lead_before_crash_min"] = None if warned.empty else float(warned.lead.median())
    return out
