import asyncio
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from hpcops import detect as D
from hpcops import rca as R
from hpcops.features import Baseline, ema
from hpcops.forecast import node_features
from hpcops.logs import TemplateVocab, parse_world_logs
from hpcops.realogs import grouping_accuracy, naive_groups
from hpcops.sim import N_NODES, N_OST, Incident, simulate, topology

ROOT = Path(__file__).resolve().parents[1]
from hpcops.logs import BIN
HAS_GO = BIN.exists()


@pytest.fixture(scope="module")
def small():
    train = [simulate(10, days=2), simulate(11, days=2)]
    test = simulate(12, days=2)
    return train, test, Baseline.fit(train)


# ---- simulation ------------------------------------------------------------------------
def test_topology_shapes():
    t = topology()
    assert len(t["switch"]) == N_NODES == 128
    assert np.bincount(t["switch"]).tolist() == [16] * 8
    for o in range(N_OST):  # every storage target has nodes in every rack
        assert set(t["rack"][t["ost"] == o]) == {0, 1, 2, 3}


def test_simulation_is_reproducible_and_down_nodes_are_nan():
    a, b = simulate(3, days=2), simulate(3, days=2)
    assert [i.start for i in a.incidents] == [i.start for i in b.incidents]
    assert np.isnan(a.X[a.down]).all()
    crashed = [i for i in a.incidents if i.crash is not None]
    assert crashed and all(a.down[i.nodes[0], i.crash] for i in crashed)


def test_heavy_jobs_run_hot_but_are_not_faults(small):
    _, w, base = small
    sig = base.signals(w)
    fault_nodes = {n for i in w.incidents if i.kind in ("fan", "dimm") for n in i.nodes}  # thermal/ECC faults only
    mask = w.heavy & ~np.isin(np.arange(N_NODES), list(fault_nodes))[:, None]
    assert np.nanpercentile(w.X[..., 1][mask], 90) > 80          # hot in raw terms
    assert np.nanpercentile(np.abs(sig["z_temp"][mask]), 99) < 6  # normal once power is accounted for


# ---- features ----------------------------------------------------------------------------
def test_ema_restarts_after_gap():
    x = np.array([[10.0, 10.0, np.nan, np.nan, 50.0, 50.0]])
    out = ema(x, 3.0)
    assert out[0, 4] == 50.0  # restarted, not dragged from 10


def test_forecast_features_do_not_use_the_future(small, tmp_path):
    if not HAS_GO:
        pytest.skip("Go binary not built")
    train, w, base = small
    parsed = parse_world_logs(w, tmp_path)
    vocab = TemplateVocab([parsed[3]])
    F1, _ = node_features(w, base, vocab.counts(w, parsed))
    cut = w.T // 2
    w2 = simulate(12, days=2)
    w2.X[:, cut + 1:] = np.random.default_rng(0).normal(50, 30, w2.X[:, cut + 1:].shape)
    w2.logs = [l for l in w2.logs if l[0] <= cut]
    parsed2 = parse_world_logs(w2, tmp_path / "b")
    F2, _ = node_features(w2, base, vocab.counts(w2, parsed2))
    np.testing.assert_allclose(np.nan_to_num(F1[:, :cut + 1]), np.nan_to_num(F2[:, :cut + 1]), rtol=1e-5, atol=1e-4)


# ---- detection ---------------------------------------------------------------------------
def test_persistence_and_episodes():
    a = np.zeros((2, 40), bool)
    a[0, 5:7] = True        # 2 minutes: filtered by 3-minute persistence
    a[1, 10:15] = True
    a[1, 25:30] = True      # gap 10 <= 15: same episode
    eps = D.episodes(D.persistent(a))
    assert eps == [(1, 12, 29)]


def test_match_and_pages():
    inc = [Incident(0, "fan", "node-1", [1], start=100, end=200, crash=200)]
    first, unmatched = D.match([(1, 150, 160), (1, 260, 270), (5, 100, 110), (6, 105, 110), (7, 300, 305)], inc)
    assert first[0] == 150
    assert len(unmatched) == 4                        # one after crash+30, three on other nodes
    assert D.pages(unmatched) == 3                    # nodes 5 and 6 start together: one page


def test_alerts_after_a_crash_do_not_count_as_detection():
    inc = [Incident(0, "dimm", "node-1", [1], start=100, end=200, crash=200)]
    first, _ = D.match([(1, 210, 215)], inc)
    assert first[0] is None


# ---- root-cause decision rule ------------------------------------------------------------
def _sig(**kw):
    base = {"temp_z_max": 0.0, "ecc_z_max": 0.0, "retrans_z_max": 0.0, "iowait_z_max": 0.0, "power_z_mean": 0.0}
    return base | kw


def test_decide_switch_and_ost_and_benign():
    st = R.OpsState.__new__(R.OpsState)
    st.topo = topology()
    sw_nodes = list(range(16, 32))
    d = R.decide_topology({}, {n: _sig(retrans_z_max=50) for n in sw_nodes}, st.topology_summary(sw_nodes), {})
    assert d == ("switch", "switch-1")
    ost_nodes = [n for n in range(128) if st.topo["ost"][n] == 2][:6]
    d = R.decide_topology({}, {n: _sig(iowait_z_max=20) for n in ost_nodes}, st.topology_summary(ost_nodes), {})
    assert d == ("ost", "ost-2")
    d = R.decide_topology({}, {7: _sig(temp_z_max=12, power_z_mean=8)}, st.topology_summary([7]), {})
    assert d == ("benign", None)          # hot because it draws more power
    d = R.decide_topology({}, {7: _sig(temp_z_max=12)}, st.topology_summary([7]), {})
    assert d == ("fan", "node-7")
    logs = {"LustreError: 11-0: <*> operation ost_write": {"count": 3, "example": "LustreError: 11-0: scratch-OST0003-osc: x"}}
    d = R.decide_topology({}, {9: _sig(iowait_z_max=20)}, st.topology_summary([9]), logs)
    assert d == ("ost", "ost-3")


# ---- real-log metric ---------------------------------------------------------------------
def test_grouping_accuracy():
    truth = ["A", "A", "B", "B", "C"]
    assert grouping_accuracy(truth, [1, 1, 2, 2, 3]) == 1.0
    assert grouping_accuracy(truth, [1, 1, 1, 1, 3]) == pytest.approx(0.2)   # A and B merged: only C right
    assert naive_groups(["job 12 done", "job 13 done"]) == ["job <*> done"] * 2


@pytest.mark.skipif(not HAS_GO, reason="Go binary not built")
def test_go_parser_extracts_bgl_content_exactly():
    from hpcops.realogs import run_go_parser
    import pandas as pd
    parsed = run_go_parser("BGL", Path(os.environ.get("TMPDIR", "/tmp")) / "hpcops_test")
    truth = pd.read_csv(ROOT / "data" / "raw" / "BGL_2k.log_structured.csv")
    assert len(parsed) == 2000
    assert (parsed.content.str.strip().values == truth.Content.astype(str).str.strip().values).all()


# ---- MCP integration ---------------------------------------------------------------------
@pytest.mark.skipif(not (ROOT / "results" / "work" / "state.pkl").exists(), reason="run hpcops.study first")
def test_mcp_server_tools_and_no_truth_leak():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "hpcops.mcp_server"], cwd=str(ROOT),
                                       env={**os.environ, "PYTHONPATH": str(ROOT)})
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                names = {t.name for t in (await s.list_tools()).tools}
                res = await s.call_tool("get_topology", {"nodes": [0, 1, 16]})
                return names, res

    names, res = asyncio.run(go())
    assert {"list_alert_groups", "get_node_signals", "get_topology", "get_node_logs"} <= names
    assert not any("truth" in n or "incident" in n for n in names)
    assert not getattr(res, "is_error", getattr(res, "isError", False))
