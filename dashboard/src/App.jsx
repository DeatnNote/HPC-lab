import React, { useEffect, useMemo, useState } from "react";

const pct = (x, d = 0) => (x == null ? "—" : `${(100 * x).toFixed(d)}%`);
const num = (x, d = 2) => (x == null ? "—" : Number(x).toFixed(d));
const mins = (x) => (x == null ? "—" : x >= 120 ? `${(x / 60).toFixed(1)} h` : `${Math.round(x)} min`);
const METHOD_LABEL = {
  static: "Fixed thresholds",
  rolling_z: "Rolling z-score (6 h)",
  iforest: "Isolation Forest",
  residual: "Load-normalised residuals",
};
const KINDS = ["fan", "dimm", "switch", "ost"];

function Tile({ label, value, note }) {
  return (
    <div className="tile">
      <div className="tile-value">{value}</div>
      <div className="tile-label">{label}</div>
      {note && <div className="tile-note">{note}</div>}
    </div>
  );
}

function LineChart({ series, height = 180, markers = [], yLabel }) {
  const w = 640, h = height, pad = { l: 44, r: 12, t: 10, b: 24 };
  const xs = series[0].t;
  const vals = series.flatMap((s) => s.y.filter((v) => v != null));
  const ymin = Math.min(...vals), ymax = Math.max(...vals);
  const x = (t) => pad.l + ((t - xs[0]) / (xs[xs.length - 1] - xs[0])) * (w - pad.l - pad.r);
  const y = (v) => h - pad.b - ((v - ymin) / (ymax - ymin || 1)) * (h - pad.t - pad.b);
  const path = (s) => {
    let d = "", pen = false;
    s.y.forEach((v, i) => {
      if (v == null) { pen = false; return; }
      d += `${pen ? "L" : "M"}${x(s.t[i]).toFixed(1)},${y(v).toFixed(1)} `;
      pen = true;
    });
    return d;
  };
  const ticks = [ymin, (ymin + ymax) / 2, ymax];
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="chart" role="img" aria-label={yLabel}>
      {ticks.map((v, i) => (
        <g key={i}>
          <line x1={pad.l} x2={w - pad.r} y1={y(v)} y2={y(v)} className="grid" />
          <text x={pad.l - 6} y={y(v) + 4} className="axis" textAnchor="end">{v.toFixed(0)}</text>
        </g>
      ))}
      {markers.map((m, i) => (
        <g key={i}>
          <line x1={x(m.t)} x2={x(m.t)} y1={pad.t} y2={h - pad.b} className={`marker ${m.cls || ""}`} />
          <text x={x(m.t) + 4} y={pad.t + 12} className="axis">{m.label}</text>
        </g>
      ))}
      {series.map((s, i) => <path key={i} d={path(s)} className={`line ${s.cls}`} fill="none" />)}
      <text x={pad.l} y={h - 6} className="axis">minute {xs[0]}</text>
      <text x={w - pad.r} y={h - 6} className="axis" textAnchor="end">minute {xs[xs.length - 1]}</text>
    </svg>
  );
}

function Legend({ items }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span key={it.label}><i className={`swatch ${it.cls}`} />{it.label}</span>
      ))}
    </div>
  );
}

export default function App() {
  const [data, setData] = useState(null);
  const [filter, setFilter] = useState("wrong");
  useEffect(() => {
    fetch("./data.json").then((r) => r.json()).then(setData);
  }, []);
  const rows = useMemo(() => {
    if (!data) return [];
    return data.agent_rows.filter((r) => (filter === "all" ? true : filter === "wrong" ? !r.ok : r.ok));
  }, [data, filter]);
  if (!data) return <div className="loading">Loading results…</div>;

  const { study, realogs, agent, example_fan: fan, example_heavy: heavy } = data;
  const det = study.detection, fc = study.forecast, rca = study.rca;
  const comp = (c) => c ?? "—";

  return (
    <main>
      <header>
        <h1>HPC Ops Lab</h1>
        <p className="sub">
          Anomaly detection, crash forecasting and root-cause analysis for a 128-node cluster. Scored on{" "}
          {study.test_days} simulated days with {study.test_incidents} injected faults (known ground truth), plus
          real LogHub supercomputer logs for parsing.
        </p>
      </header>

      <section className="tiles">
        <Tile label="Incidents detected" value={pct(det.residual.recall)} note={`${num(det.residual.false_pages_per_day)} false pages / day`} />
        <Tile label="Crashes warned ≥ 30 min ahead" value={pct(fc.lgbm.warned)} note={`median lead ${mins(fc.lgbm.median_lead_min)}`} />
        <Tile label="Root cause correct" value={pct(rca.topology.accuracy)} note={`${rca.topology.groups} alert groups, vs ${pct(rca.single_node.accuracy)} single-node`} />
        <Tile label="BGL log grouping accuracy" value={pct(realogs.parsing.BGL.drain_grouping_accuracy, 1)} note="Go parser, real logs" />
      </section>

      <section>
        <h2>Detection at the same false-page budget</h2>
        <p className="note">Thresholds set on 3 training weeks to ≤ 1 false page per day, then frozen for the test weeks.</p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Method</th><th>All</th>{KINDS.map((k) => <th key={k}>{k}</th>)}<th>False pages / day</th></tr>
            </thead>
            <tbody>
              {Object.entries(det).map(([m, r]) => (
                <tr key={m} className={m === "residual" ? "best" : ""}>
                  <td>{METHOD_LABEL[m]}</td>
                  <td>{pct(r.recall)}</td>
                  {KINDS.map((k) => <td key={k}>{pct(r.by_kind[k]?.recall)}</td>)}
                  <td>{num(r.false_pages_per_day)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="two">
        <div>
          <h3>Fan failure on {fan.incident.component}</h3>
          <LineChart
            yLabel="temperature"
            series={[{ t: fan.t, y: fan.temp, cls: "a" }]}
            markers={[{ t: fan.incident.start, label: "fault", cls: "warn" }, { t: fan.incident.crash, label: "crash", cls: "bad" }]}
          />
          <Legend items={[{ label: "CPU temperature (°C)", cls: "a" }]} />
        </div>
        <div>
          <h3>Heavy job on healthy {heavy.node}</h3>
          <LineChart yLabel="temperature" series={[{ t: heavy.t, y: heavy.temp, cls: "a" }]} />
          <Legend items={[{ label: "CPU temperature (°C)", cls: "a" }]} />
        </div>
        <p className="note span">
          Both run hot. The heavy job draws more power to get there; the failing fan does not. Comparing
          temperature with power drawn is what separates them.
        </p>
      </section>

      <section>
        <h2>Crash forecasting</h2>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Method</th><th>Crashes warned</th><th>Median lead</th><th>Fan lead</th><th>DIMM lead</th><th>False alarms / day</th></tr></thead>
            <tbody>
              {[["rule", "Residual rule"], ["lgbm", "LightGBM + log templates"]].map(([k, label]) => (
                <tr key={k}>
                  <td>{label}</td><td>{pct(fc[k].warned)} of {fc[k].crashes}</td><td>{mins(fc[k].median_lead_min)}</td>
                  <td>{mins(fc[k].by_kind.fan.median_lead_min)}</td><td>{mins(fc[k].by_kind.dimm.median_lead_min)}</td>
                  <td>{num(fc[k].false_alarms_per_day)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="note">PR-AUC for "crash in the next 2 hours": {num(fc.pr_auc_lgbm)} (LightGBM) vs {num(fc.pr_auc_rule)} (rule); base rate {pct(fc.positive_rate, 2)}.</p>
      </section>

      <section>
        <h2>Root cause</h2>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Investigator</th><th>Overall</th>{["fan", "dimm", "switch", "ost", "benign"].map((k) => <th key={k}>{k}</th>)}</tr></thead>
            <tbody>
              {[["single_node", "First alerting node only"], ["topology", "Topology + logs"]].map(([k, label]) => (
                <tr key={k} className={k === "topology" ? "best" : ""}>
                  <td>{label}</td><td>{pct(rca[k].accuracy)}</td>
                  {["fan", "dimm", "switch", "ost", "benign"].map((t) => (
                    <td key={t}>{rca[k].by_truth[t] ? `${pct(rca[k].by_truth[t].accuracy)} (${rca[k].by_truth[t].n})` : "—"}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="note">
          Agent run over MCP ({agent.policy} policy): {pct(agent.accuracy, 1)} on {agent.groups} groups, {agent.tool_calls} tool calls,{" "}
          {agent.tool_errors} errors, median tool latency {num(agent.tool_latency_ms_p50)} ms. The LLM policy is implemented but was not run.
        </p>
        <div className="filters">
          {["wrong", "right", "all"].map((f) => (
            <button key={f} className={filter === f ? "on" : ""} onClick={() => setFilter(f)}>{f}</button>
          ))}
        </div>
        <div className="table-wrap scroll">
          <table>
            <thead><tr><th>Week</th><th>Group</th><th>Truth</th><th>Diagnosis</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={`${r.world}-${r.group}`} className={r.ok ? "" : "miss"}>
                  <td>{r.world}</td><td>{r.group}</td>
                  <td>{r.truth[0]} · {comp(r.truth[1])}</td><td>{r.pred[0]} · {comp(r.pred[1])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2>Real logs (LogHub, 2,000 lines each)</h2>
        <div className="table-wrap">
          <table>
            <thead><tr><th>System</th><th>True templates</th><th>Go parser</th><th>Naive digit mask</th></tr></thead>
            <tbody>
              {Object.entries(realogs.parsing).map(([k, r]) => (
                <tr key={k}><td>{k}</td><td>{r.true_templates}</td><td>{pct(r.drain_grouping_accuracy, 1)}</td><td>{pct(r.naive_grouping_accuracy, 1)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="note">
          BGL alerts with a template-lookup detector: {pct(realogs.alerts.BGL.random_split_mean.recall)} recall on a random split,{" "}
          {pct(realogs.alerts.BGL.time_split.recall)} on a time-ordered split — all {realogs.alerts.BGL.time_split.test_alerts} later
          alerts used templates never seen earlier.
        </p>
      </section>
      <footer>Simulated telemetry with injected faults; LogHub samples are real. Not production data.</footer>
    </main>
  );
}
