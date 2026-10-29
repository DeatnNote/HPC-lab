"""Node-crash forecasting: will this node crash in the next 2 hours?

Samples: every up node every 5 minutes. Features: load-normalised residuals and their trends,
ECC counts over 30/120 minutes, utilisation, and trailing-60-minute counts of every log template
the Go parser found (the model is not told which templates matter).

Compared with a rule on the same residual signals (the detector's temperature and ECC z-scores).
Both are scored as alarm episodes: a crash counts as warned if an alarm on that node starts 30 to
720 minutes before it; an alarm episode with no crash on that node in the next 12 hours is false.
Each method's threshold is set on the training weeks to <= `budget` false alarms per day.
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score

from .detect import episodes, persistent
from .features import Baseline, rolling_sum, ECC, UTIL
from .logs import TemplateVocab
from .sim import MINUTES_PER_DAY, World

HORIZON, STRIDE, MIN_LEAD, MAX_LEAD = 120, 5, 30, 720


def node_features(w: World, base: Baseline, tcounts: np.ndarray) -> tuple[np.ndarray, list[str]]:
    sig = base.signals(w)
    X = w.X
    zt = sig["z_temp"]
    zt_lag = np.full_like(zt, np.nan); zt_lag[:, 30:] = zt[:, :-30]
    ecc30 = sig["ecc30"]
    ecc120 = rolling_sum(X[..., ECC], 120)
    ecc_prev = np.zeros_like(ecc30); ecc_prev[:, 30:] = ecc30[:, :-30]
    feats = [zt, zt - zt_lag, sig["z_pow"], np.log1p(ecc30), np.log1p(ecc120),
             np.log1p(ecc30) - np.log1p(ecc_prev), X[..., UTIL], sig["z_ret"], sig["z_iow"]]
    names = ["z_temp", "z_temp_trend30", "z_power", "log_ecc30", "log_ecc120", "ecc_growth30", "util", "z_retrans", "z_iowait"]
    F = np.concatenate([np.stack(feats, -1), tcounts], -1)
    return F, names + [f"tmpl_{i}" for i in range(tcounts.shape[-1])]


def crash_labels(w: World) -> np.ndarray:
    y = np.zeros((w.X.shape[0], w.T), bool)
    for inc in w.incidents:
        if inc.crash is not None:
            n = inc.nodes[0]
            y[n, max(0, inc.crash - HORIZON):inc.crash] = True
    return y


def samples(w, base, vocab, parsed):
    F, names = node_features(w, base, vocab.counts(w, parsed))
    y = crash_labels(w)
    up = np.isfinite(w.X[..., 0])
    t_idx = np.arange(0, w.T, STRIDE)
    Fs, ys = F[:, t_idx], y[:, t_idx]
    ok = up[:, t_idx]
    return Fs[ok], ys[ok], names


def fit_model(train_sets, seed=0, neg_frac=0.1):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for F, y, _ in train_sets:
        keep = y | (rng.random(len(y)) < neg_frac)
        Xs.append(F[keep]); ys.append(y[keep])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    return model.fit(X, y)


def full_scores(w, base, vocab, parsed, model=None):
    """Minute-level score (N, T): model probability, or the rule max(z_temp, z_ecc)."""
    if model is None:
        sig = base.signals(w)
        return np.nan_to_num(np.fmax(sig["z_temp"], sig["z_ecc"]))
    F, _ = node_features(w, base, vocab.counts(w, parsed))
    N, T, D = F.shape
    s = model.predict_proba(F.reshape(-1, D))[:, 1].reshape(N, T)
    s[~np.isfinite(w.X[..., 0])] = 0
    return s


def alarm_eval(scores, worlds, thr):
    warned, leads, n_crash, false, days = 0, [], 0, 0, 0.0
    for s, w in zip(scores, worlds):
        eps = episodes(persistent(s > thr))
        crashes = {}
        for inc in w.incidents:
            if inc.crash is not None:
                crashes.setdefault(inc.nodes[0], []).append(inc.crash)
        for n, cl in crashes.items():
            for c in cl:
                n_crash += 1
                starts = [a for nn, a, _ in eps if nn == n and c - MAX_LEAD <= a <= c - MIN_LEAD]
                if starts:
                    warned += 1
                    leads.append(c - min(starts))
        for n, a, _ in eps:
            if not any(a <= c <= a + MAX_LEAD for c in crashes.get(n, [])):
                false += 1
        days += w.T / MINUTES_PER_DAY
    return {"crashes": n_crash, "warned": warned / max(n_crash, 1),
            "median_lead_min": float(np.median(leads)) if leads else None, "false_alarms_per_day": false / days}


def calibrate(scores, worlds, budget=1.0):
    flat = np.concatenate([s[s > 0].ravel() for s in scores])
    grid = np.unique(np.quantile(flat, 1 - np.geomspace(0.2, 1e-6, 80)))
    for thr in grid:
        if alarm_eval(scores, worlds, thr)["false_alarms_per_day"] <= budget:
            return float(thr)
    return float(flat.max())


def pr_auc(model, test_sets):
    X = np.concatenate([F for F, _, _ in test_sets]); y = np.concatenate([y for _, y, _ in test_sets])
    return float(average_precision_score(y, model.predict_proba(X)[:, 1])), float(y.mean())


def rule_pr_auc(worlds, base):
    ss, ys = [], []
    for w in worlds:
        s = np.nan_to_num(np.fmax(base.signals(w)["z_temp"], base.signals(w)["z_ecc"]))
        y = crash_labels(w)
        up = np.isfinite(w.X[..., 0])
        t = np.arange(0, w.T, STRIDE)
        ss.append(s[:, t][up[:, t]]); ys.append(y[:, t][up[:, t]])
    return float(average_precision_score(np.concatenate(ys), np.concatenate(ss)))
