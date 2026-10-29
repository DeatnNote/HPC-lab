"""MCP server exposing read-only cluster-operations tools over stdio.

    python -m hpcops.mcp_server

On start it rebuilds the test weeks from results/work/state.pkl (written by `python -m hpcops.study`):
re-simulates the seeds, re-parses their logs with the Go parser, and re-derives the residual
detector's alert groups. The ground truth is never exposed through a tool.
"""
from __future__ import annotations

import pickle
from pathlib import Path

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from .logs import parse_world_logs
from .rca import OpsState
from .sim import simulate

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "results" / "work"


def load_state() -> OpsState:
    cfg = pickle.loads((WORK / "state.pkl").read_bytes())
    worlds = [simulate(s) for s in cfg["test_seeds"]]
    parsed = {w.seed: parse_world_logs(w, WORK / "mcp") for w in worlds}
    return OpsState(worlds, cfg["base"], cfg["threshold"], parsed)


STATE: OpsState | None = None
server = _Server("hpc-ops")


def state() -> OpsState:
    global STATE
    if STATE is None:
        STATE = load_state()
    return STATE


@server.tool()
def list_worlds() -> list[int]:
    """List the monitored cluster-weeks (seeds) available for investigation."""
    return sorted(state().worlds)


@server.tool()
def list_alert_groups(world: int) -> list[dict]:
    """List alert groups (concurrent anomaly episodes) for one cluster-week: id, start/end minute, node count."""
    return state().list_groups(world)


@server.tool()
def get_alert_group(world: int, group_id: int) -> dict:
    """Return the alerting nodes and time window of one alert group."""
    return state().get_group(world, group_id)


@server.tool()
def get_node_signals(world: int, node: int, start: int, end: int) -> dict:
    """Load-normalised z-scores for one node over a window: max temperature residual (given power),
    ECC errors vs background, retransmits and I/O wait (given utilisation), and mean power residual."""
    return state().node_signals(world, node, start, end)


@server.tool()
def get_topology(nodes: list[int]) -> dict:
    """Count the given nodes by network switch, storage target (OST) and rack."""
    return state().topology_summary(nodes)


@server.tool()
def get_node_logs(world: int, node: int, start: int, end: int) -> dict:
    """Most frequent log templates (from the Go parser) on a node from 60 minutes before start to end."""
    return state().node_logs(world, node, start, end)


if __name__ == "__main__":
    state()  # build before accepting requests
    server.run("stdio")
