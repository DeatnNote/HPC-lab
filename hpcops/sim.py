"""Simulated supercomputer telemetry with known injected faults (ground truth for every incident).

Cluster: 4 racks x 2 network switches x 16 nodes = 128 nodes; 4 parallel-filesystem storage targets
(OSTs) striped across racks, so an OST fault shows up on nodes in every rack. One simulated week at
1-minute resolution per seed.

Per-node metrics: power (W), CPU temperature (C), correctable ECC errors/min, network retransmits/min,
I/O wait (%), job utilisation.

Faults (each with a component, start, end and optional crash time):
  fan    (node)   cooling loss: temperature climbs without a power rise -> throttling -> crash
  dimm   (node)   correctable ECC errors grow exponentially for hours -> uncorrectable error -> crash
  switch (switch) link flapping: retransmit bursts on all 16 nodes behind the switch
  ost    (ost)    slow storage target: I/O wait rises on the *busy* nodes that use it (idle nodes look fine)
Benign confounder: "heavy" jobs (vector-heavy codes) draw ~20% more power and run hot across up to
64 nodes. They are normal operation, and are where naive temperature thresholds fire.

Each node also emits syslog-style lines from fixed templates (background noise plus fault-related
messages); these go through the Go parser like real logs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

N_RACKS, SW_PER_RACK, NODES_PER_SW, N_OST = 4, 2, 16, 4
N_NODES = N_RACKS * SW_PER_RACK * NODES_PER_SW
MINUTES_PER_DAY = 1440
METRICS = ["power", "temp", "ecc", "retrans", "iowait", "util"]


def topology() -> dict:
    node = np.arange(N_NODES)
    switch = node // NODES_PER_SW
    rack = switch // SW_PER_RACK
    ost = (node // 2) % N_OST  # striped: every OST has nodes in every rack
    return {"rack": rack, "switch": switch, "ost": ost}


@dataclass
class Incident:
    id: int
    kind: str          # fan | dimm | switch | ost
    component: str     # e.g. "node-17", "switch-3", "ost-2"
    nodes: list        # affected nodes
    start: int
    end: int
    crash: int | None = None   # minute the node went down (fan / dimm only)


@dataclass
class World:
    seed: int
    T: int
    X: np.ndarray            # (N_NODES, T, len(METRICS)) float32; NaN while a node is down
    heavy: np.ndarray        # (N_NODES, T) bool: node running a heavy (benign hot) job
    down: np.ndarray         # (N_NODES, T) bool
    incidents: list = field(default_factory=list)
    logs: list = field(default_factory=list)   # (minute, node, text)

    def incident_dicts(self):
        return [asdict(i) for i in self.incidents]


LOG_BACKGROUND = [
    (0.004, "RAS KERNEL INFO instruction cache parity error corrected"),
    (0.002, "kernel: nfs: server {srv} OK"),
    (0.003, "sshd[{pid}]: session opened for user {user}"),
    (0.0015, "slurmd[{pid}]: launch task {job}.{step} request from {uid}@10.1.{a}.{b}"),
    (0.0008, "ntpd[{pid}]: synchronized to 10.0.0.{a}, stratum 2"),
    (0.0006, "kernel: EDAC MC0: 1 CE memory read error on CPU_SrcID#0_Ha#0_Chan#{a}_DIMM#{b}"),
    (0.0004, "LustreError: {pid}:0:(client.c:2116:ptlrpc_expire_one_request()) @@@ Request sent has timed out for slow reply: {a}s"),
    (0.0003, "kernel: mlx5_core 0000:{a}:00.0: Port module event: module {b}, Cable plugged"),
]


def _fmt(rng, tpl: str) -> str:
    return tpl.format(srv=f"nfs{rng.integers(1, 4)}", pid=int(rng.integers(1000, 65000)),
                      user=f"u{rng.integers(100, 999)}", job=int(rng.integers(100000, 999999)),
                      step=int(rng.integers(0, 9)), uid=int(rng.integers(1000, 5000)),
                      a=int(rng.integers(1, 250)), b=int(rng.integers(0, 16)))


def _jobs(rng, T):
    """Return util (N,T), heavy (N,T)."""
    util = np.full((N_NODES, T), 0.05, np.float32)
    heavy = np.zeros((N_NODES, T), bool)
    t = 0
    while True:
        t += int(rng.exponential(35))
        if t >= T:
            break
        size = int(rng.choice([8, 16, 32, 64], p=[0.4, 0.3, 0.2, 0.1]))
        start_node = int(rng.integers(0, N_NODES // size)) * size
        dur = int(np.clip(rng.lognormal(np.log(150), 0.6), 30, 900))
        sl = (slice(start_node, start_node + size), slice(t, min(T, t + dur)))
        is_heavy = rng.random() < 0.15
        util[sl] = np.maximum(util[sl], rng.uniform(0.85, 0.98))
        if is_heavy:
            heavy[sl] = True
    util += rng.normal(0, 0.02, util.shape).astype(np.float32)
    return np.clip(util, 0, 1), heavy


def simulate(seed: int, days: int = 7, n_fan=6, n_dimm=6, n_switch=4, n_ost=4) -> World:
    rng = np.random.default_rng(seed)
    T = days * MINUTES_PER_DAY
    topo = topology()
    util, heavy = _jobs(rng, T)
    t_axis = np.arange(T)

    # Physical baseline.
    intensity = np.where(heavy, 1.2, 1.0)
    power = 120 + 230 * util * intensity + rng.normal(0, 5, util.shape)
    inlet = 22 + 1.5 * np.sin(2 * np.pi * (t_axis / MINUTES_PER_DAY - 0.3))[None, :] + \
        np.array([0.0, 0.8, -0.5, 1.2])[topo["rack"]][:, None]
    temp_extra = np.zeros_like(util)
    ecc_rate = np.full(util.shape, 0.02)
    retrans_extra = np.zeros_like(util)
    iowait_extra = np.zeros_like(util)
    down = np.zeros(util.shape, bool)
    throttle = np.ones(util.shape)
    incidents: list[Incident] = []
    fault_logs: list = []

    busy_until = {}  # component -> minute it is free again (avoid overlapping faults on one component)

    def pick_start(comp, lo=120, margin=900):
        for _ in range(100):
            s = int(rng.integers(lo, T - margin))
            if busy_until.get(comp, -1) < s:
                return s
        return None

    kinds = ["fan"] * n_fan + ["dimm"] * n_dimm + ["switch"] * n_switch + ["ost"] * n_ost
    for kind in kinds:
        if kind in ("fan", "dimm"):
            n = int(rng.integers(0, N_NODES))
            comp = f"node-{n}"
            s = pick_start(comp, margin=1400)
            if s is None:
                continue
            nodes = [n]
            if kind == "fan":
                rate = rng.uniform(0.06, 0.16)  # C per minute at full load
                end = min(T, s + 600)
                ramp = np.cumsum(rate * (0.3 + util[n, s:end]))
                temp_extra[n, s:end] += ramp
                crash = None
                for k in range(s, end):
                    est = inlet[n, k] + 0.17 * power[n, k] + temp_extra[n, k]
                    if est > 92:
                        throttle[n, k] = 0.5
                    if est > 97:
                        crash = k
                        break
                if crash is not None:
                    temp_extra[n, crash:s + len(ramp)] -= ramp[crash - s:]  # cooling restored on reboot
                end = crash if crash is not None else end
                for k in range(s, end):
                    if rng.random() < 0.08:
                        fault_logs.append((k, n, f"ipmi: fan FAN{rng.integers(1, 7)} speed {rng.integers(300, 900)} RPM below lower critical threshold"))
            else:
                tau = rng.uniform(40, 110)
                end_max = min(T, s + 1400)
                lam = 0.02 * np.exp((np.arange(end_max - s)) / tau)
                over = np.nonzero(lam > 60)[0]
                crash = s + int(over[0]) if len(over) else None
                end = crash if crash is not None else end_max
                ecc_rate[n, s:end] = np.maximum(ecc_rate[n, s:end], lam[: end - s])
            if crash is not None:
                down_end = min(T, crash + int(rng.integers(60, 240)))
                down[n, crash:down_end] = True
                fault_logs.append((crash, n, "kernel: Kernel panic - not syncing: Fatal machine check" if kind == "dimm"
                                   else "kernel: CPU0: Package temperature above threshold, cpu clock throttled; shutting down"))
                busy_until[comp] = down_end + 60
            else:
                busy_until[comp] = end + 60
            incidents.append(Incident(len(incidents), kind, comp, nodes, s, end, crash))
        elif kind == "switch":
            sw = int(rng.integers(0, N_RACKS * SW_PER_RACK))
            comp = f"switch-{sw}"
            s = pick_start(comp)
            if s is None:
                continue
            end = min(T, s + int(rng.integers(20, 60)))
            nodes = np.nonzero(topo["switch"] == sw)[0]
            for k in range(s, end):
                if rng.random() < 0.45:
                    retrans_extra[nodes, k] += rng.poisson(350, len(nodes))
                    for nn in nodes:
                        if rng.random() < 0.15:
                            fault_logs.append((k, int(nn), f"kernel: mlx5_core 0000:{rng.integers(1, 99)}:00.0 ib0: link down, retraining (CRC errors {rng.integers(10, 900)})"))
            busy_until[comp] = end + 60
            incidents.append(Incident(len(incidents), kind, comp, nodes.tolist(), s, end))
        else:  # ost
            o = int(rng.integers(0, N_OST))
            comp = f"ost-{o}"
            s = pick_start(comp)
            if s is None:
                continue
            end = min(T, s + int(rng.integers(30, 120)))
            nodes = np.nonzero(topo["ost"] == o)[0]
            iowait_extra[nodes, s:end] += 28 * util[nodes, s:end] * rng.uniform(0.8, 1.2, (len(nodes), 1))
            for k in range(s, end):
                for nn in nodes:
                    if util[nn, k] > 0.5 and rng.random() < 0.03:
                        fault_logs.append((k, int(nn), f"LustreError: 11-0: scratch-OST{o:04d}-osc: operation ost_write to node 10.2.0.{rng.integers(1, 250)} failed: rc = -107"))
            busy_until[comp] = end + 60
            incidents.append(Incident(len(incidents), kind, comp, nodes.tolist(), s, end))

    # Assemble metrics.
    util_eff = util * throttle
    power = 120 + 230 * util_eff * intensity + rng.normal(0, 5, util.shape)
    # temperature follows power with a first-order lag (tau = 5 min)
    target = inlet + 0.17 * power + temp_extra
    temp = np.empty_like(target)
    temp[:, 0] = target[:, 0]
    a = 1 / 5
    for k in range(1, T):
        temp[:, k] = temp[:, k - 1] + a * (target[:, k] - temp[:, k - 1])
    temp += rng.normal(0, 0.6, temp.shape)
    ecc = rng.poisson(ecc_rate)
    retrans = rng.poisson(2 + 20 * util_eff) + retrans_extra
    iowait = np.clip(util_eff * 3 + rng.normal(0, 0.8, util.shape) + iowait_extra, 0, 100)

    X = np.stack([power, temp, ecc, retrans, iowait, util_eff], axis=-1).astype(np.float32)
    X[down] = np.nan

    # Logs: background + fault messages (+ ECC-driven EDAC lines), nothing from down nodes.
    logs = []
    for p, tpl in LOG_BACKGROUND:
        hits = np.argwhere(rng.random((N_NODES, T)) < p)
        for n, k in hits:
            if not down[n, k]:
                logs.append((int(k), int(n), _fmt(rng, tpl)))
    for n, k in np.argwhere(ecc >= 3):
        if not down[n, k] and rng.random() < 0.5:
            logs.append((int(k), int(n), f"kernel: EDAC MC0: {int(ecc[n, k])} CE memory read error on CPU_SrcID#0_Ha#0_Chan#{rng.integers(0, 4)}_DIMM#{rng.integers(0, 3)}"))
    for n, k in np.argwhere((temp > 85) & ~down):
        if rng.random() < 0.05:
            logs.append((int(k), int(n), f"kernel: CPU{rng.integers(0, 64)}: Core temperature above threshold, cpu clock throttled (total events = {rng.integers(1, 9999)})"))
    logs.extend(fault_logs)
    logs.sort()
    return World(seed, T, X, heavy, down, incidents, logs)


if __name__ == "__main__":
    w = simulate(0)
    print(len(w.incidents), "incidents", len(w.logs), "log lines")
    for i in w.incidents:
        print(i.kind, i.component, i.start, i.end, i.crash, len(i.nodes))
    print("temp p99 normal / heavy:", np.nanpercentile(w.X[..., 1][~w.heavy], 99), np.nanpercentile(w.X[..., 1][w.heavy], 99))
