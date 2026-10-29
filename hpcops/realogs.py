"""Real-log evaluation on LogHub supercomputer samples (BGL, HPC, Thunderbird; 2,000 lines each).

1. Parsing: runs the Go Drain parser on the raw logs and scores grouping accuracy against
   LogHub's hand-labelled event IDs, next to a naive "mask every token containing a digit" baseline.
2. Alert detection on BGL (the only one of the three 2k samples with alert lines; the Thunderbird
   sample's 2,000 lines are all labelled normal): a time-ordered split
   (first 60% to learn, last 40% to test) versus a random split, to show how much a random split flatters
   a template-lookup detector.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BIN = ROOT / "bin" / ("logparse.exe" if os.name == "nt" else "logparse")

DATASETS = {
    "BGL": dict(format="<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
                masks=[r"core\.\d+"], st=0.5, depth=4),
    "HPC": dict(format="<LogId> <Node> <Component> <State> <Time> <Flag> <Content>",
                masks=[r"=\d+"], st=0.5, depth=4),
    "Thunderbird": dict(format="<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>: <Content>",
                        masks=[], st=0.5, depth=4),
}


def run_go_parser(name: str, out_dir: Path) -> pd.DataFrame:
    cfg = DATASETS[name]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}_parsed.jsonl"
    cmd = [str(BIN), "parse", "-in", str(RAW / f"{name}_2k.log"), "-out", str(out),
           "-templates", str(out_dir / f"{name}_templates.json"), "-format", cfg["format"],
           "-st", str(cfg["st"]), "-depth", str(cfg["depth"])]
    for m in cfg["masks"]:
        cmd += ["-mask", m]
    subprocess.run(cmd, check=True, capture_output=True)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    return pd.DataFrame({"line": [r["line"] for r in rows], "content": [r["content"] for r in rows],
                         "cluster_id": [r["cluster_id"] for r in rows],
                         "fields": [r["fields"] for r in rows]})


def grouping_accuracy(truth, pred) -> float:
    """Share of lines whose predicted group contains exactly the same lines as their true group
    (the standard LogHub parsing metric)."""
    truth, pred = np.asarray(truth), np.asarray(pred)
    df = pd.DataFrame({"t": truth, "p": pred})
    ok = 0
    for _, g in df.groupby("p"):
        t_ids = g["t"].unique()
        if len(t_ids) == 1 and (truth == t_ids[0]).sum() == len(g):
            ok += len(g)
    return ok / len(df)


def naive_groups(contents) -> list:
    return [" ".join("<*>" if re.search(r"\d", t) else t for t in c.split()) for c in contents]


def parsing_study(out_dir: Path) -> dict:
    res = {}
    for name in DATASETS:
        parsed = run_go_parser(name, out_dir)
        truth = pd.read_csv(RAW / f"{name}_2k.log_structured.csv")
        if len(parsed) != len(truth):
            raise RuntimeError(f"{name}: parsed {len(parsed)} lines, truth has {len(truth)}")
        content_match = float(np.mean([a.strip() == str(b).strip() for a, b in zip(parsed.content, truth.Content)]))
        res[name] = {
            "lines": int(len(parsed)),
            "true_templates": int(truth.EventId.nunique()),
            "drain_templates": int(parsed.cluster_id.nunique()),
            "drain_grouping_accuracy": grouping_accuracy(truth.EventId, parsed.cluster_id),
            "naive_grouping_accuracy": grouping_accuracy(truth.EventId, naive_groups(parsed.content)),
            "content_extraction_match": content_match,
        }
    return res


def _template_detector(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Flag a test line as an alert if its template was mostly alerts in training.
    Templates never seen in training are not flagged (the lookup has no evidence)."""
    rate = train.groupby("cluster_id")["alert"].mean()
    pred = test["cluster_id"].map(rate).fillna(0.0).values >= 0.5
    y = test["alert"].values.astype(bool)
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    unseen = ~test["cluster_id"].isin(rate.index)
    return {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "test_alerts": int(y.sum()),
            "alerts_on_unseen_templates": int((y & unseen.values).sum())}


def _rule(test: pd.DataFrame, pred) -> dict:
    y = test["alert"].values.astype(bool)
    pred = np.asarray(pred, bool)
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    return {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}


def alert_study(out_dir: Path, seeds=range(20)) -> dict:
    res = {}
    for name in ("BGL",):
        parsed = run_go_parser(name, out_dir)
        truth = pd.read_csv(RAW / f"{name}_2k.log_structured.csv")
        df = pd.DataFrame({"cluster_id": parsed.cluster_id.values, "alert": (truth.Label != "-").values,
                           "ts": truth.Timestamp.values,
                           "level": [f.get("Level", "") for f in parsed.fields]}).sort_values("ts", kind="stable").reset_index(drop=True)
        cut = int(0.6 * len(df))
        time_split = _template_detector(df.iloc[:cut], df.iloc[cut:])
        rand = []
        for s in seeds:
            idx = np.random.default_rng(s).permutation(len(df))
            rand.append(_template_detector(df.iloc[idx[:cut]], df.iloc[idx[cut:]]))
        train, test = df.iloc[:cut], df.iloc[cut:]
        fatal = (test.level == "FATAL").values
        new_tmpl = (~test.cluster_id.isin(train.cluster_id)).values
        res[name] = {
            "alert_share": float(df.alert.mean()),
            "time_split": time_split,
            "time_split_severity_rule": _rule(test, fatal),
            "time_split_new_template_and_fatal": _rule(test, fatal & new_tmpl),
            "time_split_new_template_or_known_alert": _rule(
                test, new_tmpl | (test.cluster_id.map(train.groupby("cluster_id")["alert"].mean()).fillna(0).values >= 0.5)),
            "random_split_mean": {k: float(np.mean([r[k] for r in rand])) for k in ("precision", "recall")},
            "random_split_seeds": len(rand),
        }
    return res


def main():
    out_dir = ROOT / "results" / "realogs"
    res = {"parsing": parsing_study(out_dir), "alerts": alert_study(out_dir)}
    (ROOT / "results" / "realogs.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
