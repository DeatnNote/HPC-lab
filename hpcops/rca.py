"""Root-cause analysis of alert groups, scored against the injected faults.

Alert episodes from the residual detector are grouped (episodes starting within 15 minutes of the
group's first episode). One fault can therefore produce several groups (e.g. storage slowness reaching
different busy nodes at different times); each group is scored on its own. Each group must be explained as one of
  fan / dimm (component = node), switch (component = switch), ost (component = storage target),
  or benign (no fault).
A diagnosis is correct only if both the fault type and the component are right.

Two investigators use the same read-only tools (also served over MCP in mcp_server.py):
  single_node  looks only at the first alerting node: its strongest signal names the fault and the
               node is blamed
  topology     looks at every alerting node, finds the shared switch or storage target, checks
               whether a hot node is simply drawing more power (a heavy job), and reads the node's
               logs for a matching message
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .detect import episodes, persistent, score_residual
from .features import Baseline
from .sim import N_OST, NODES_PER_SW, World, topology

SIGNALS = {"temp": "z_temp", "ecc": "z_ecc", "retrans": "z_ret", "iowait": "z_iow"}
KIND_OF = {"temp": "fan", "ecc": "dimm", "retrans": "switch", "iowait": "ost"}
LOG_HINTS = {"fan": ["fan", "temperature above threshold"], "dimm": ["EDAC", "machine check"],
             "switch": ["link down", "retraining"], "ost": ["LustreError: 11-0"]}


@dataclass
class Group:
    world: int
    gid: int
    start: int
    end: int
    nodes: list


class OpsState:
    """Read-only view of the monitored cluster that the tools expose."""

    def __init__(self, worlds: list[World], base: Baseline, thr: float, parsed: dict | None = None):
        self.worlds = {w.seed: w for w in worlds}
        self.base = base
        self.topo = topology()
        self.sig = {w.seed: base.signals(w) for w in worlds}
        self.parsed = parsed or {}
        self.groups: dict[tuple[int, int], Group] = {}
        for w in worlds:
            eps = sorted(episodes(persistent(score_residual(w, base) > thr)), key=lambda e: e[1])
            gid, cur = 0, None
            for n, a, b in eps:
                if cur is None or a - cur.start > 15:
                    cur = Group(w.seed, gid, a, b, [])
                    self.groups[(w.seed, gid)] = cur
                    gid += 1
                if n not in cur.nodes:
                    cur.nodes.append(n)
                cur.end = max(cur.end, b)

    # ---- tools -------------------------------------------------------------------------
    def list_groups(self, world: int) -> list[dict]:
        return [{"group_id": g.gid, "start": g.start, "end": g.end, "n_nodes": len(g.nodes)}
                for (wd, _), g in sorted(self.groups.items()) if wd == world]

    def get_group(self, world: int, group_id: int) -> dict:
        g = self.groups[(world, group_id)]
        return {"group_id": g.gid, "start": g.start, "end": g.end, "nodes": list(map(int, g.nodes))}

    def node_signals(self, world: int, node: int, start: int, end: int) -> dict:
        s = self.sig[world]
        sl = slice(max(0, start), max(start + 1, min(end + 1, start + 30)))
        out = {}
        for name, key in SIGNALS.items():
            v = s[key][node, sl]
            out[f"{name}_z_max"] = float(np.nanmax(v)) if np.isfinite(v).any() else None
        v = s["z_pow"][node, sl]
        out["power_z_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else None
        return out

    def topology_summary(self, nodes: list[int]) -> dict:
        nodes = list(map(int, nodes))
        sw = Counter(int(self.topo["switch"][n]) for n in nodes)
        ost = Counter(int(self.topo["ost"][n]) for n in nodes)
        rack = Counter(int(self.topo["rack"][n]) for n in nodes)
        return {"n_nodes": len(nodes),
                "switches": {f"switch-{k}": v for k, v in sw.most_common()},
                "osts": {f"ost-{k}": v for k, v in ost.most_common()},
                "racks": {f"rack-{k}": v for k, v in rack.most_common()},
                "nodes_per_switch": NODES_PER_SW, "nodes_per_ost": int(len(self.topo["ost"]) // N_OST)}

    def node_logs(self, world: int, node: int, start: int, end: int, before: int = 60) -> dict:
        """Top templates on the node with their count and the most recent raw line."""
        if world not in self.parsed:
            return {}
        minute, nd, cid, templates, contents = self.parsed[world]
        text = {t["id"]: t["template"] for t in templates}
        m = np.flatnonzero((nd == node) & (minute >= start - before) & (minute <= end))
        counts = Counter(text[cid[i]] for i in m)
        latest = {text[cid[i]]: contents[i] for i in m}  # most recent raw line per template
        return {t: {"count": c, "example": latest[t]} for t, c in counts.most_common(8)}

    # ---- ground truth (not exposed as a tool) -------------------------------------------
    def truth(self, world: int, group_id: int) -> tuple[str, str | None]:
        g = self.groups[(world, group_id)]
        votes = Counter()
        for inc in self.worlds[world].incidents:
            impact = inc.crash if inc.crash is not None else inc.end
            overlap = [n for n in g.nodes if n in inc.nodes]
            if overlap and g.start <= impact + 30 and g.end >= inc.start:
                votes[(inc.kind, inc.component)] += len(overlap)
        return votes.most_common(1)[0][0] if votes else ("benign", None)


def dominant(signals: dict) -> tuple[str, float]:
    best, val = None, -np.inf
    for name in SIGNALS:
        v = signals.get(f"{name}_z_max")
        if v is not None and v > val:
            best, val = name, v
    return best, val


def diagnose_single(state: OpsState, world: int, group_id: int) -> tuple[str, str | None]:
    g = state.get_group(world, group_id)
    n = g["nodes"][0]
    sig_name, _ = dominant(state.node_signals(world, n, g["start"], g["end"]))
    return KIND_OF[sig_name], f"node-{n}"


def decide_topology(group: dict, signals: dict[int, dict], topo: dict, logs: dict) -> tuple[str, str | None]:
    """Pure decision rule over tool outputs (shared by the direct investigator and the MCP agent)."""
    totals = Counter()
    for s in signals.values():
        for name in SIGNALS:
            v = s.get(f"{name}_z_max")
            if v is not None and v > 0:
                totals[name] += v
    if not totals:
        return "benign", None
    top = totals.most_common(1)[0][0]
    if top == "retrans":
        sw, cnt = next(iter(topo["switches"].items()))
        if cnt >= 4:
            return "switch", sw
    if top == "iowait":
        ost, cnt = next(iter(topo["osts"].items()))
        if cnt >= 3 and cnt >= 0.8 * topo["n_nodes"]:
            return "ost", ost
        # Too few busy nodes to triangulate: a Lustre error naming the storage target confirms it.
        for t, info in logs.items():
            m = re.search(r"OST(\d{4})", f"{t} {info.get('example', '') if isinstance(info, dict) else ''}")
            if m:
                return "ost", f"ost-{int(m.group(1))}"
        if any("LustreError: 11-0" in t for t in logs) and topo["n_nodes"] <= 2:
            return "ost", ost
    # node-level fault: pick the node with the strongest signal of that type
    node = max(signals, key=lambda n: signals[n].get(f"{top}_z_max") or -np.inf)
    if top == "temp":
        p = signals[node].get("power_z_mean") or 0.0
        fan_log = any("fan" in t for t in logs)
        if p > 3 and not fan_log:
            return "benign", None  # hot because it is drawing more power: a heavy job, not a fault
        return "fan", f"node-{node}"
    if top in ("retrans", "iowait"):
        return "benign", None  # a lone node with network or I/O noise and no shared component
    return "dimm", f"node-{node}"


def diagnose_topology(state: OpsState, world: int, group_id: int, max_nodes: int = 32) -> tuple[str, str | None]:
    g = state.get_group(world, group_id)
    nodes = g["nodes"][:max_nodes]
    signals = {n: state.node_signals(world, n, g["start"], g["end"]) for n in nodes}
    topo = state.topology_summary(g["nodes"])
    top_node = max(signals, key=lambda n: dominant(signals[n])[1])
    logs = state.node_logs(world, top_node, g["start"], g["end"])
    return decide_topology(g, signals, topo, logs)


def score(state: OpsState, diagnose) -> dict:
    rows = []
    for (w, gid) in sorted(state.groups):
        truth = state.truth(w, gid)
        pred = diagnose(state, w, gid)
        rows.append({"world": w, "group": gid, "truth": truth, "pred": tuple(pred),
                     "kind_ok": truth[0] == pred[0], "ok": tuple(truth) == tuple(pred)})
    by = {}
    for r in rows:
        k = r["truth"][0]
        by.setdefault(k, []).append(r["ok"])
    return {"groups": len(rows), "accuracy": float(np.mean([r["ok"] for r in rows])),
            "kind_accuracy": float(np.mean([r["kind_ok"] for r in rows])),
            "by_truth": {k: {"n": len(v), "accuracy": float(np.mean(v))} for k, v in sorted(by.items())},
            "rows": rows}
