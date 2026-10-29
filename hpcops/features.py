"""Per node-minute signals shared by detection, forecasting and root-cause analysis.

The key idea is load normalisation: compare each metric to what the node's own load predicts
(temperature given power drawn, retransmits and I/O wait given utilisation), using a baseline
fitted on training weeks. A hot node running a power-hungry job is then normal; a hot node that
is *not* drawing extra power is not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sim import METRICS, MINUTES_PER_DAY, World, topology

P, TEMP, ECC, RET, IOW, UTIL = range(len(METRICS))


def ema(x: np.ndarray, tau: float) -> np.ndarray:
    """Exponential moving average along time (axis 1). NaN (node down) holds the last value, and the
    average restarts from the first reading after a gap, so a reboot does not drag in stale state."""
    out = np.empty_like(x)
    a = 1.0 / tau
    prev = np.nan_to_num(x[:, 0])
    was_down = np.isnan(x[:, 0])
    for k in range(x.shape[1]):
        cur = x[:, k]
        down = np.isnan(cur)
        restart = was_down & ~down
        prev = np.where(down, prev, np.where(restart, cur, prev + a * (cur - prev)))
        out[:, k] = prev
        was_down = down
    return out


def rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing window sum along axis 1 (NaN treated as 0)."""
    c = np.cumsum(np.nan_to_num(x), axis=1)
    out = c.copy()
    out[:, w:] = c[:, w:] - c[:, :-w]
    return out


def _design(w: World):
    X = w.X
    topo = topology()
    T = w.T
    tod = 2 * np.pi * (np.arange(T) / MINUTES_PER_DAY)
    p_lag = ema(X[..., P], 5.0)
    cols = [p_lag, np.broadcast_to(np.sin(tod), (X.shape[0], T)), np.broadcast_to(np.cos(tod), (X.shape[0], T))]
    for r in range(1, 4):
        cols.append(np.broadcast_to((topo["rack"] == r)[:, None].astype(float), (X.shape[0], T)))
    cols.append(np.ones((X.shape[0], T)))
    return np.stack(cols, -1)


@dataclass
class Baseline:
    """Load-normalised expectations fitted on training worlds."""
    beta_temp: np.ndarray
    sd_temp: float
    beta_ret: np.ndarray
    sd_ret5: float
    beta_iow: np.ndarray
    sd_iow5: float
    beta_pow: np.ndarray
    sd_pow: float
    ecc_rate: float

    @staticmethod
    def fit(worlds: list[World]) -> "Baseline":
        def robust_sd(r):
            r = r[np.isfinite(r)]
            return float(1.4826 * np.median(np.abs(r - np.median(r))))

        Ds, temps, utils, rets, iows, pows, eccs = [], [], [], [], [], [], []
        for w in worlds:
            D = _design(w)
            ok = np.isfinite(w.X[..., TEMP])
            Ds.append(D[ok]); temps.append(w.X[..., TEMP][ok])
            utils.append(w.X[..., UTIL][ok]); rets.append(w.X[..., RET][ok])
            iows.append(w.X[..., IOW][ok]); pows.append(w.X[..., P][ok]); eccs.append(w.X[..., ECC][ok])
        D = np.concatenate(Ds); temp = np.concatenate(temps); u = np.concatenate(utils)
        ret = np.concatenate(rets); iow = np.concatenate(iows); pw = np.concatenate(pows)
        # Robust-ish fit: least squares, refit after dropping the largest 1% of residuals (fault minutes).
        def fit_ls(A, y):
            b = np.linalg.lstsq(A, y, rcond=None)[0]
            r = y - A @ b
            keep = np.abs(r) < np.quantile(np.abs(r), 0.99)
            return np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
        bt = fit_ls(D, temp)
        Au = np.stack([u, np.ones_like(u)], 1)
        br, bi, bp = fit_ls(Au, ret), fit_ls(Au, iow), fit_ls(Au, pw)
        base = Baseline(bt, robust_sd(temp - D @ bt), br, 1.0, bi, 1.0, bp, robust_sd(pw - Au @ bp), 0.0)
        # 5-minute-mean residual spreads, measured on the same worlds
        r5r, r5i = [], []
        for w in worlds:
            s = base.signals(w, standardise=False)
            r5r.append(s["ret5"][np.isfinite(s["ret5"])]); r5i.append(s["iow5"][np.isfinite(s["iow5"])])
        base.sd_ret5 = robust_sd(np.concatenate(r5r))
        base.sd_iow5 = robust_sd(np.concatenate(r5i))
        allecc = np.concatenate(eccs)
        base.ecc_rate = float(np.mean(allecc[allecc < 3]))
        return base

    def signals(self, w: World, standardise: bool = True) -> dict:
        X = w.X
        D = _design(w)
        temp_r = X[..., TEMP] - D @ self.beta_temp
        u = X[..., UTIL]
        ret_r = X[..., RET] - (self.beta_ret[0] * u + self.beta_ret[1])
        iow_r = X[..., IOW] - (self.beta_iow[0] * u + self.beta_iow[1])
        pow_r = X[..., P] - (self.beta_pow[0] * u + self.beta_pow[1])
        ret5 = rolling_sum(ret_r, 5) / 5
        iow5 = rolling_sum(iow_r, 5) / 5
        ecc30 = rolling_sum(X[..., ECC], 30)
        down = ~np.isfinite(X[..., TEMP])
        for a in (ret5, iow5):
            a[down] = np.nan
        out = {"temp_r": temp_r, "ret5": ret5, "iow5": iow5, "pow_r": pow_r, "ecc30": ecc30, "down": down}
        if standardise:
            mu = 30 * self.ecc_rate
            out["z_temp"] = ema(temp_r, 3.0) / self.sd_temp
            out["z_ret"] = ret5 / self.sd_ret5
            out["z_iow"] = iow5 / self.sd_iow5
            out["z_ecc"] = (ecc30 - mu) / np.sqrt(mu + 1.0)
            out["z_pow"] = pow_r / self.sd_pow
            for k in ("z_temp", "z_ret", "z_iow", "z_ecc", "z_pow"):
                out[k][down] = np.nan
        return out
