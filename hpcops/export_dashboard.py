"""Bundle results into dashboard/public/data.json for the React dashboard."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from .features import TEMP, P
from .sim import simulate

ROOT = Path(__file__).resolve().parents[1]


def _series(w, base, n, a, b, step=2):
    sig = base.signals(w)
    t = list(range(a, b, step))
    clean = lambda v: [None if not np.isfinite(x) else round(float(x), 2) for x in v]
    return {"t": t, "temp": clean(w.X[n, a:b:step, TEMP]), "power": clean(w.X[n, a:b:step, P]),
            "z_temp": clean(sig["z_temp"][n, a:b:step]), "heavy": [bool(x) for x in w.heavy[n, a:b:step]]}


def main():
    res = json.loads((ROOT / "results" / "study.json").read_text())
    real = json.loads((ROOT / "results" / "realogs.json").read_text())
    agent = json.loads((ROOT / "results" / "agent_playbook.json").read_text())
    cfg = pickle.loads((ROOT / "results" / "work" / "state.pkl").read_bytes())
    w = simulate(cfg["test_seeds"][0])
    fan = next(i for i in w.incidents if i.kind == "fan" and i.crash is not None)
    n = fan.nodes[0]
    example = {"incident": {"kind": "fan", "component": fan.component, "start": fan.start, "crash": fan.crash},
               **_series(w, cfg["base"], n, max(0, fan.start - 240), min(w.T, fan.crash + 60))}
    # a heavy job on a healthy node for contrast
    healthy = [m for m in range(w.X.shape[0]) if all(m not in i.nodes for i in w.incidents if i.kind in ("fan", "dimm"))]
    hn, ht = next((m, int(np.flatnonzero(w.heavy[m])[0])) for m in healthy if w.heavy[m].any())
    heavy = _series(w, cfg["base"], hn, max(0, ht - 120), min(w.T, ht + 300))
    heavy["node"] = f"node-{hn}"
    rows = [{"world": r["world"], "group": r["group"], "truth": r["truth"], "pred": r["pred"], "ok": r["ok"]}
            for r in agent["rows"]]
    out = {"study": res, "realogs": real, "agent": {k: v for k, v in agent.items() if k != "rows"},
           "agent_rows": rows, "example_fan": example, "example_heavy": heavy}
    dest = ROOT / "dashboard" / "public" / "data.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, default=float))
    print("wrote", dest, dest.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
