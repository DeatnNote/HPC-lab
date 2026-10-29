"""Root-cause agent that investigates alert groups through the MCP server.

    python -m hpcops.agent                    # deterministic playbook policy (what the results use)
    python -m hpcops.agent --backend anthropic  # LLM policy; needs ANTHROPIC_API_KEY (not run for the results)

The playbook policy follows the same steps an on-call engineer would (group -> per-node signals ->
topology -> logs) and applies rca.decide_topology to what the tools return. The LLM policy gets the
same MCP tools plus a `submit_diagnosis` tool and decides for itself which to call.
Every tool call is traced (tool, arguments, latency, ok) to results/agent_trace.jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .rca import decide_topology, dominant

ROOT = Path(__file__).resolve().parents[1]
KINDS = ["fan", "dimm", "switch", "ost", "benign"]


def _payload(result):
    sc = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
    if sc is not None:
        return sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc
    text = "".join(getattr(c, "text", "") for c in result.content)
    return json.loads(text) if text else None


class Tools:
    def __init__(self, session: ClientSession, trace: list):
        self.session, self.trace = session, trace

    async def __call__(self, name: str, **args):
        t = time.perf_counter()
        ok = True
        try:
            res = await self.session.call_tool(name, args)
            ok = not getattr(res, "is_error", getattr(res, "isError", False))
            return _payload(res)
        except Exception:
            ok = False
            raise
        finally:
            self.trace.append({"tool": name, "args": args, "ms": round(1000 * (time.perf_counter() - t), 2), "ok": ok})


class PlaybookPolicy:
    name = "playbook"

    async def investigate(self, tools: Tools, world: int, group_id: int):
        g = await tools("get_alert_group", world=world, group_id=group_id)
        signals = {}
        for n in g["nodes"][:32]:
            signals[n] = await tools("get_node_signals", world=world, node=n, start=g["start"], end=g["end"])
        topo = await tools("get_topology", nodes=g["nodes"])
        top_node = max(signals, key=lambda n: dominant(signals[n])[1])
        logs = await tools("get_node_logs", world=world, node=top_node, start=g["start"], end=g["end"])
        return decide_topology(g, signals, topo, logs or {})


SYSTEM = """You are an on-call engineer for a 128-node supercomputer (4 racks x 2 switches x 16 nodes,
4 storage targets striped across racks). Diagnose one alert group. Use the tools to look at the
alerting nodes' load-normalised signals, their topology and their logs, then call submit_diagnosis
with kind in {fan, dimm, switch, ost, benign} and component (node-N, switch-N, ost-N, or null for benign).
A node that is hot because it is drawing extra power is running a heavy job, which is benign."""


class AnthropicPolicy:
    """LLM policy over the same MCP tools. Implemented but not run for the published results."""
    name = "anthropic"

    def __init__(self, model: str | None = None, max_steps: int = 30):
        import anthropic  # optional dependency
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("HPCOPS_MODEL", "claude-sonnet-5-5")
        self.max_steps = max_steps
        self.usage = {"api_calls": 0, "input_tokens": 0, "output_tokens": 0}

    async def investigate(self, tools: Tools, world: int, group_id: int):
        listed = await tools.session.list_tools()
        specs = [{"name": t.name, "description": t.description or "",
                  "input_schema": getattr(t, "input_schema", None) or t.inputSchema} for t in listed.tools]
        specs.append({"name": "submit_diagnosis", "description": "Submit the final diagnosis.",
                      "input_schema": {"type": "object", "properties": {
                          "kind": {"type": "string", "enum": KINDS},
                          "component": {"type": ["string", "null"]}}, "required": ["kind", "component"]}})
        msgs = [{"role": "user", "content": f"Diagnose alert group {group_id} in world {world}."}]
        for _ in range(self.max_steps):
            resp = self.client.messages.create(model=self.model, max_tokens=1024, system=SYSTEM, tools=specs, messages=msgs)
            self.usage["api_calls"] += 1
            u = getattr(resp, "usage", None)
            if u is not None:
                self.usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
                self.usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0
            msgs.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                if block.name == "submit_diagnosis":
                    return block.input["kind"], block.input.get("component")
                out = await tools(block.name, **block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(out)})
            if not results:
                break
            msgs.append({"role": "user", "content": results})
        return "benign", None


async def run(policy, limit: int | None = None) -> dict:
    from .mcp_server import load_state  # ground truth comes from a separate in-process copy, never the server

    truth_state = load_state()
    params = StdioServerParameters(command=sys.executable, args=["-m", "hpcops.mcp_server"], cwd=str(ROOT),
                                   env={**os.environ, "PYTHONPATH": str(ROOT)})
    trace, rows = [], []
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = Tools(session, trace)
            worlds = await tools("list_worlds")
            for world in worlds:
                for g in await tools("list_alert_groups", world=world):
                    if limit is not None and len(rows) >= limit:
                        break
                    pred = tuple(await policy.investigate(tools, world, g["group_id"]))
                    truth = truth_state.truth(world, g["group_id"])
                    rows.append({"world": world, "group": g["group_id"], "pred": pred, "truth": truth,
                                 "ok": tuple(truth) == pred})
    (ROOT / "results" / "agent_trace.jsonl").write_text("".join(json.dumps(t) + "\n" for t in trace))
    lat = sorted(t["ms"] for t in trace)
    return {"policy": policy.name, "groups": len(rows), "accuracy": sum(r["ok"] for r in rows) / max(len(rows), 1),
            "tool_calls": len(trace), "tool_errors": sum(not t["ok"] for t in trace),
            "tool_latency_ms_p50": lat[len(lat) // 2] if lat else None,
            "tool_latency_ms_p95": lat[int(0.95 * (len(lat) - 1))] if lat else None,
            "model": getattr(policy, "model", None), "usage": getattr(policy, "usage", None), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["playbook", "anthropic"], default="playbook")
    ap.add_argument("--limit", type=int, default=None, help="investigate only the first N alert groups")
    ap.add_argument("--model", default=None, help="Anthropic model id (default: $HPCOPS_MODEL or claude-sonnet-5-5)")
    a = ap.parse_args()
    policy = PlaybookPolicy() if a.backend == "playbook" else AnthropicPolicy(model=a.model)
    res = asyncio.run(run(policy, a.limit))
    suffix = "" if a.limit is None else f"_limit{a.limit}"
    (ROOT / "results" / f"agent_{policy.name}{suffix}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
