"""Run the full simulated study: detection, crash forecasting, root-cause analysis.

Training weeks: seeds 0-2. Test weeks: seeds 3-7 (thresholds and models frozen before testing).
Writes results/study.json and results/work/state.pkl (what the MCP server needs to rebuild the cluster view).
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

from . import detect as D
from . import forecast as F
from . import rca as R
from .features import Baseline
from .logs import TemplateVocab, parse_world_logs
from .sim import simulate

ROOT = Path(__file__).resolve().parents[1]
TRAIN, TEST = [0, 1, 2], [3, 4, 5, 6, 7]
WORK = ROOT / "results" / "work"


def build(train_seeds=TRAIN, test_seeds=TEST):
    train = [simulate(s) for s in train_seeds]
    test = [simulate(s) for s in test_seeds]
    base = Baseline.fit(train)
    return train, test, base


def main():
    t0 = time.time()
    train, test, base = build()
    out = {"train_seeds": TRAIN, "test_seeds": TEST,
           "test_incidents": int(sum(len(w.incidents) for w in test)),
           "test_days": float(sum(w.T for w in test) / 1440)}
    kinds = {}
    for w in test:
        for i in w.incidents:
            kinds[i.kind] = kinds.get(i.kind, 0) + 1
    out["test_incidents_by_kind"] = kinds

    # 1. Detection
    ifm = D.fit_iforest(train)
    det, thresholds = {}, {}
    for m in D.METHODS:
        tr = [D.SCORERS[m](w, base, model=ifm) for w in train]
        thr = D.calibrate(tr, train, budget=1.0)
        _, fp_train = D.evaluate_threshold(tr, train, thr)
        te = [D.SCORERS[m](w, base, model=ifm) for w in test]
        df, fp = D.evaluate_threshold(te, test, thr)
        det[m] = D.summarise(df, fp) | {"threshold": thr, "train_false_pages_per_day": fp_train}
        thresholds[m] = thr
        print(f"[detect] {m}: recall {det[m]['recall']:.3f} fp/day {fp:.2f}  ({time.time()-t0:.0f}s)")
    out["detection"] = det

    # 2. Forecasting
    ptr = {w.seed: parse_world_logs(w, WORK) for w in train}
    pte = {w.seed: parse_world_logs(w, WORK) for w in test}
    vocab = TemplateVocab([p[3] for p in ptr.values()])
    trs = [F.samples(w, base, vocab, ptr[w.seed]) for w in train]
    tes = [F.samples(w, base, vocab, pte[w.seed]) for w in test]
    model = F.fit_model(trs)
    ap, base_rate = F.pr_auc(model, tes)
    fc = {"pr_auc_lgbm": ap, "pr_auc_rule": F.rule_pr_auc(test, base), "positive_rate": base_rate,
          "templates": vocab.templates}
    for name, mdl in (("rule", None), ("lgbm", model)):
        st = [F.full_scores(w, base, vocab, ptr[w.seed], mdl) for w in train]
        thr = F.calibrate(st, train)
        se = [F.full_scores(w, base, vocab, pte[w.seed], mdl) for w in test]
        fc[name] = F.alarm_eval(se, test, thr) | {"threshold": thr, "train": F.alarm_eval(st, train, thr)}
        # lead by fault kind
        by = {}
        for kind in ("fan", "dimm"):
            sub = [type(w)(w.seed, w.T, w.X, w.heavy, w.down, [i for i in w.incidents if i.kind == kind], w.logs) for w in test]
            r = F.alarm_eval(se, sub, thr)
            by[kind] = {"crashes": r["crashes"], "warned": r["warned"], "median_lead_min": r["median_lead_min"]}
        fc[name]["by_kind"] = by
        print(f"[forecast] {name}: {fc[name]}  ({time.time()-t0:.0f}s)")
    imp = sorted(zip(model.feature_importances_.tolist(), F.node_features(test[0], base, vocab.counts(test[0], pte[3]))[1]), reverse=True)
    fc["top_features"] = [{"feature": n, "importance": int(v)} for v, n in imp[:8]]
    out["forecast"] = fc

    # 3. Root-cause analysis on the residual detector's alert groups
    state = R.OpsState(test, base, thresholds["residual"], pte)
    res = {}
    for name, fn in (("single_node", R.diagnose_single), ("topology", R.diagnose_topology)):
        s = R.score(state, fn)
        res[name] = {k: v for k, v in s.items() if k != "rows"}
        print(f"[rca] {name}: acc {s['accuracy']:.3f} on {s['groups']} groups  ({time.time()-t0:.0f}s)")
    out["rca"] = res

    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / "state.pkl", "wb") as f:
        pickle.dump({"base": base, "threshold": thresholds["residual"], "test_seeds": TEST}, f)
    (ROOT / "results" / "study.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
