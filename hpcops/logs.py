"""Send simulated node logs through the Go parser and build per-node template-count features."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from .sim import World

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin" / ("logparse.exe" if os.name == "nt" else "logparse")
FORMAT = "<Minute> <Node> <Content>"


def parse_world_logs(w: World, work: Path) -> tuple:
    """Return (minute, node, cluster_id, templates, contents), using the Go binary."""
    work.mkdir(parents=True, exist_ok=True)
    raw = work / f"sim_logs_seed{w.seed}.log"
    raw.write_text("".join(f"{k} node-{n} {text}\n" for k, n, text in w.logs))
    out, tmpl = work / f"sim_logs_seed{w.seed}.jsonl", work / f"sim_templates_seed{w.seed}.json"
    subprocess.run([str(BIN), "parse", "-in", str(raw), "-out", str(out), "-templates", str(tmpl),
                    "-format", FORMAT], check=True, capture_output=True)
    minute, node, cid, content = [], [], [], []
    for line in out.read_text().splitlines():
        r = json.loads(line)
        minute.append(int(r["fields"]["Minute"]))
        node.append(int(r["fields"]["Node"].split("-")[1]))
        cid.append(r["cluster_id"])
        content.append(r["content"])
    return np.array(minute), np.array(node), np.array(cid), json.loads(tmpl.read_text()), np.array(content, dtype=object)


class TemplateVocab:
    """Maps template *text* to a feature column, so ids from separate parser runs line up."""

    def __init__(self, templates_per_world: list[list[dict]], min_count: int = 5):
        counts = {}
        for ts in templates_per_world:
            for t in ts:
                counts[t["template"]] = counts.get(t["template"], 0) + t["count"]
        self.templates = sorted(t for t, c in counts.items() if c >= min_count)
        self.index = {t: i for i, t in enumerate(self.templates)}

    def counts(self, w: World, parsed, window: int = 60) -> np.ndarray:
        """(N, T, V) trailing-window counts of each template per node."""
        minute, node, cid, templates = parsed[:4]
        id2col = {t["id"]: self.index.get(t["template"], -1) for t in templates}
        V = len(self.templates)
        C = np.zeros((w.X.shape[0], w.T, V), np.float32)
        cols = np.array([id2col.get(c, -1) for c in cid])
        ok = cols >= 0
        np.add.at(C, (node[ok], minute[ok], cols[ok]), 1)
        cs = np.cumsum(C, axis=1)
        cs[:, window:] = cs[:, window:] - cs[:, :-window].copy()
        return cs
