"""Interactive HTML visualizations for AI World sweep runs.

Consumes a sweep directory (runs/<condition>/<config-hash>/ containing seed_* run
dirs) and renders one plotly figure per chart into a target directory:

    # Every chart, default sweep discovery + ./visualizations/
    python scripts/visualize_analysis.py --sweep runs/<cond>/<hash> --all

    # Individual charts
    python scripts/visualize_analysis.py --sweep <dir> --survival
    python scripts/visualize_analysis.py --sweep <dir> --dispersion
    python scripts/visualize_analysis.py --sweep <dir> --heatmap
    python scripts/visualize_analysis.py --sweep <dir> --completion
    python scripts/visualize_analysis.py --sweep <dir> --network --seed 0
    python scripts/visualize_analysis.py --sweep <dir> --vocabulary

Zero LLM cost -- every chart reads ticks.jsonl / events.jsonl / status.json
directly. Requires plotly + networkx (see requirements.txt).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import plotly.graph_objects as go  # noqa: E402

ACTION_KINDS = ["consume", "move", "talk", "gather", "transfer", "build",
                "take", "attack", "wait"]
KIND_COLORS = {
    "consume": "#2b83ba", "move": "#abdda4", "talk": "#fdae61",
    "gather": "#d7191c", "transfer": "#7b3294", "build": "#a6d96a",
    "take": "#6a51a3", "attack": "#e41a1c", "wait": "#969696",
}


def _seed_dirs(sweep_dir: str) -> list[str]:
    dirs = sorted(glob.glob(os.path.join(sweep_dir, "seed_*")))
    dirs = [d for d in dirs if os.path.isdir(d)]

    def seed_key(path: str) -> int:
        name = os.path.basename(path)
        digits = "".join(c for c in name if c.isdigit())
        return int(digits) if digits else 0

    return sorted(dirs, key=seed_key)


def _load_jsonl(path: str) -> list[dict]:
    records = []
    if not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_tick_records(run_dir: str) -> list[dict]:
    return _load_jsonl(os.path.join(run_dir, "ticks.jsonl"))


def _load_event_records(run_dir: str) -> list[dict]:
    return _load_jsonl(os.path.join(run_dir, "events.jsonl"))


def _load_status(run_dir: str) -> dict:
    return _load_json(os.path.join(run_dir, "status.json"))


def _seed_label(run_dir: str) -> str:
    return os.path.basename(run_dir).replace("seed_", "seed ")


def _action_kind(action: str) -> str:
    if not action:
        return "wait"
    if action in ("consume", "gather", "build", "wait"):
        return action
    for prefix in ("move", "talk", "transfer", "take", "attack"):
        if action.startswith(prefix + ":"):
            return prefix
    return "wait"


def _talk_target(action: str) -> str | None:
    if action.startswith("talk:") and ":" in action:
        return action.split(":", 1)[1]
    return None


def survival_figure(sweep: str) -> go.Figure:
    dirs = _seed_dirs(sweep)
    fig = go.Figure()
    hover = []
    for run_dir in dirs:
        seed_label = _seed_label(run_dir)
        ticks = _load_tick_records(run_dir)
        status = _load_status(run_dir)
        horizon = int(status.get("expected_ticks", 60)) if status else 60
        agents: dict[str, int] = {}
        deaths: dict[str, int] = {}
        for rec in ticks:
            aid = rec["agent_id"]
            if aid not in agents:
                agents[aid] = rec["tick"]
            if rec.get("alive"):
                agents[aid] = rec["tick"]
            if rec.get("cause_of_death"):
                deaths[aid] = rec["tick"]
        for aid in sorted(agents):
            start = agents[aid]
            end = deaths.get(aid, horizon)
            color = "#d91a1a" if aid in deaths else "#2e8b57"
            fig.add_trace(go.Bar(
                name=f"{seed_label} · {aid}",
                y=[1],
                x=[end - start + 1],
                orientation="h",
                base=start - 1,
                marker=dict(color=color),
                hovertemplate=(f"<b>{seed_label} · {aid}</b><br>"
                               f"birth tick: {start}<br>"
                               f"{'death tick: %{x}<br>' if aid in deaths else ''}"
                               f"alive: {aid not in deaths}<extra></extra>"),
                showlegend=False,
            ))
            hover.append((seed_label, aid, start, end, aid in deaths))
    fig.update_layout(
        title="Agent survival timelines across seeds",
        xaxis_title="Tick",
        yaxis_title="",
        barmode="overlay",
        showlegend=False,
        height=max(500, 22 * len(hover)),
        template="plotly_white",
        yaxis=dict(visible=False),
    )
    return fig


def dispersion_figure(sweep: str, window: int = 5) -> go.Figure:
    """Mean pairwise behavioral distance per tick block, averaged across seeds.

    Uses the same action-mix distance as aiworld.analysis.behavioral_metrics
    .behavioral_divergence_over_time, but aggregated over the whole sweep so the
    headline primary-outcome trajectory is visible in one interactive view.
    """
    dirs = _seed_dirs(sweep)
    from aiworld.analysis.behavioral_metrics import behavioral_divergence_over_time

    per_seed: list[tuple[list[int], list[float]]] = []
    max_tick = 0
    for run_dir in dirs:
        div = behavioral_divergence_over_time(run_dir, window=window)
        if not div:
            continue
        ticks = sorted(div)
        per_seed.append((ticks, [div[t] for t in ticks]))
        max_tick = max(max_tick, ticks[-1])

    bin_ends = list(range(window, max_tick + 1, window))
    means = {}
    sds = {}
    counts = {}
    for ticks, vals in per_seed:
        for t, v in zip(ticks, vals):
            means.setdefault(t, []).append(v)
    for t in bin_ends:
        vals = means.get(t, [])
        if not vals:
            continue
        counts[t] = vals
        means[t] = float(np.mean(vals))
        sds[t] = float(np.std(vals))

    xs = sorted(means)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs,
        y=[means[t] + sds[t] for t in xs],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=xs,
        y=[means[t] - sds[t] for t in xs],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(43,131,186,0.2)",
        name="±1 SD",
    ))
    fig.add_trace(go.Scatter(
        x=xs,
        y=[means[t] for t in xs],
        mode="lines+markers",
        line=dict(color="#2b83ba", width=2),
        marker=dict(size=6),
        name="mean pairwise behavioral distance",
        hovertemplate="tick %{x}<br>dispersion %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Behavioral dispersion over time (window={window}) — mean ± SD across {len(per_seed)} seeds",
        xaxis_title="Tick",
        yaxis_title="Mean pairwise distance (action-mix, standardized)",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def death_heatmap(sweep: str) -> go.Figure:
    dirs = _seed_dirs(sweep)
    status0 = _load_status(dirs[0]) if dirs else {}
    horizon = int(status0.get("expected_ticks", 60)) if status0 else 60
    n_ticks = horizon + 1
    labels = []
    rows = np.zeros((len(dirs), n_ticks), dtype=int)
    for i, run_dir in enumerate(dirs):
        labels.append(_seed_label(run_dir))
        for ev in _load_event_records(run_dir):
            if ev.get("kind") != "death":
                continue
            t = int(ev.get("tick", 0))
            if 0 <= t < n_ticks:
                rows[i, t] += 1
    z = rows[:, 1:]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=list(range(1, n_ticks)),
        y=labels,
        colorscale=[[0, "#f7fbff"], [0.5, "#74add1"], [1, "#d73027"]],
        zmin=0,
        colorbar=dict(title="deaths"),
        hovertemplate="seed %{y}<br>tick %{x}<br>deaths %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Agent death events by seed and tick",
        xaxis_title="Tick",
        yaxis_title="Seed",
        template="plotly_white",
        height=max(500, len(labels) * 22),
    )
    return fig


def completion_figure(sweep: str) -> go.Figure:
    dirs = _seed_dirs(sweep)
    x = [_seed_label(d) for d in dirs]
    observed = []
    expected = []
    colors = []
    for d in dirs:
        status = _load_status(d)
        exp = int(status.get("expected_ticks", 60))
        obs = int(status.get("observed_ticks", 0))
        observed.append(obs)
        expected.append(exp)
        colors.append("#2e8b57" if obs >= exp else "#d91a1a")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x, y=observed,
        marker=dict(color=colors),
        name="observed ticks",
        hovertemplate="%{x}<br>observed ticks: %{y}<extra></extra>",
    ))
    fig.add_hline(
        y=expected[0], line=dict(color="#333", dash="dash", width=1),
        annotation_text=f"expected ({expected[0]})",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Seed completion — observed vs expected ticks",
        xaxis_title="Seed",
        yaxis_title="Ticks",
        template="plotly_white",
        showlegend=False,
        height=max(450, len(dirs) * 22),
    )
    return fig


def vocabulary_figure(sweep: str) -> go.Figure:
    dirs = _seed_dirs(sweep)
    counts: dict[int, dict[str, int]] = {}
    alive: dict[int, int] = {}
    for run_dir in dirs:
        for rec in _load_tick_records(run_dir):
            if not rec.get("alive", False):
                continue
            t = int(rec["tick"])
            kind = _action_kind(rec.get("proposed_action") or rec.get("chosen_action"))
            counts.setdefault(t, {}).setdefault(kind, 0)
            counts[t][kind] += 1
            alive[t] = alive.get(t, 0) + 1

    xs = sorted(counts)
    fig = go.Figure()
    for kind in ACTION_KINDS:
        fracs = [counts[t].get(kind, 0) / max(1, alive[t]) for t in xs]
        fig.add_trace(go.Scatter(
            x=xs, y=fracs, mode="lines",
            line=dict(color=KIND_COLORS[kind], width=1.5),
            stackgroup="one", groupnorm="fraction",
            name=kind,
            hovertemplate=f"{kind}: %{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        title="Action vocabulary mix over time (fraction of alive agents)",
        xaxis_title="Tick",
        yaxis_title="Fraction of agents",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def interaction_network(sweep: str, seed_dir: str, endpoint: int | None = None) -> go.Figure:
    import networkx as nx

    def cumulative_graph(ticks: list[dict], upto: int) -> nx.DiGraph:
        graph = nx.DiGraph()
        for rec in ticks:
            if rec["tick"] > upto:
                break
            src = rec["agent_id"]
            graph.add_node(src)
            target = _talk_target(rec.get("proposed_action") or rec.get("chosen_action"))
            opp = rec.get("pre_action_opportunities") or {}
            co = opp.get("co_located_agents") or []
            if target is not None:
                if graph.has_edge(src, target):
                    graph[src][target]["weight"] += 1
                else:
                    graph.add_edge(src, target, weight=1)
            for other in co:
                graph.add_node(other)
                a, b = (src, other) if src < other else (other, src)
                if graph.has_edge(a, b):
                    graph[a][b]["weight"] += 1
                else:
                    graph.add_edge(a, b, weight=1)
        return graph

    def traces_for(graph: nx.DiGraph) -> list[go.Scatter]:
        edge_x, edge_y, edge_hover = [], [], []
        for u, v, data in graph.edges(data=True):
            label = f"{u} → {v}  ×{data.get('weight', 1)}"
            edge_x.extend([node_pos[u][0], node_pos[v][0], None])
            edge_y.extend([node_pos[u][1], node_pos[v][1], None])
            edge_hover.extend([label, label, None])
        node_x = [node_pos[n][0] for n in graph.nodes()]
        node_y = [node_pos[n][1] for n in graph.nodes()]
        node_size = [14 + 6 * graph.degree(n) for n in graph.nodes()]
        node_hover = [f"{n}<br>degree {graph.degree(n)}" for n in graph.nodes()]
        return [
            go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(color="#999", width=1.5),
                hoverinfo="skip", showlegend=False,
            ),
            go.Scatter(
                x=node_x, y=node_y, mode="markers+text",
                text=[n for n in graph.nodes()],
                textposition="top center",
                marker=dict(size=node_size, color="#2b83ba"),
                customdata=node_hover,
                hovertemplate="%{customdata}<extra></extra>",
                showlegend=False,
            ),
        ]

    ticks = sorted(_load_tick_records(seed_dir), key=lambda r: r["tick"])
    if not ticks:
        raise SystemExit(f"No ticks in {seed_dir}")
    max_tick = max(r["tick"] for r in ticks)
    if endpoint is not None:
        max_tick = min(max_tick, endpoint)
    full = cumulative_graph(ticks, 10**9)
    pos = nx.spring_layout(full, seed=42, k=0.8)
    node_pos = {n: pos[n] for n in full.nodes()}

    frames = []
    slider_steps = []
    for upto in range(1, max_tick + 1):
        gi = cumulative_graph(ticks, upto)
        frames.append(go.Frame(data=traces_for(gi), name=str(upto)))
        slider_steps.append(dict(
            label=str(upto),
            method="animate",
            args=[[str(upto)],
                  dict(mode="immediate", frame=dict(duration=0, redraw=True),
                       transition=dict(duration=0))],
        ))

    fig = go.Figure(data=traces_for(cumulative_graph(ticks, 1)), frames=frames)
    fig.update_layout(
        title=f"Agent interaction network — {os.path.basename(seed_dir)} "
              f"(talk + co-location edges, cumulative)",
        template="plotly_white",
        width=900,
        height=700,
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="up to tick "),
            pad=dict(t=50),
            steps=slider_steps,
        )],
        xaxis=dict(visible=False, showgrid=False),
        yaxis=dict(visible=False, showgrid=False),
    )
    return fig


DEFAULT_OUTDIR = os.path.join(os.path.dirname(__file__), "..", "visualizations")


def _write(fig: go.Figure, outdir: str, name: str):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.write_html(path, include_plotlyjs="inline", full_html=True)
    print(f"  wrote {os.path.relpath(path)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", help="sweep dir containing seed_* run dirs")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR,
                    help="output directory (default: visualizations/)")
    ap.add_argument("--window", type=int, default=5,
                    help="block size for behavioral dispersion (default 5)")
    ap.add_argument("--seed", default="seed_0",
                    help="seed subdir to render the interaction network for (default seed_0)")
    ap.add_argument("--endpoint", type=int, default=None,
                    help="cap the network at this tick (default: last tick)")
    ap.add_argument("--all", action="store_true", help="render every chart")
    ap.add_argument("--survival", action="store_true")
    ap.add_argument("--dispersion", action="store_true")
    ap.add_argument("--heatmap", action="store_true")
    ap.add_argument("--completion", action="store_true")
    ap.add_argument("--network", action="store_true")
    ap.add_argument("--vocabulary", action="store_true")
    args = ap.parse_args()

    any_chart = any([args.all, args.survival, args.dispersion, args.heatmap,
                     args.completion, args.network, args.vocabulary])
    if not args.sweep or not any_chart:
        ap.print_help()
        raise SystemExit(1)

    dirs = _seed_dirs(args.sweep)
    if not dirs:
        raise SystemExit(f"No seed_* run dirs found under {args.sweep}")

    os.makedirs(args.outdir, exist_ok=True)

    if args.all or args.completion:
        print("rendering seed completion...")
        _write(completion_figure(args.sweep), args.outdir, "seed_completion.html")
    if args.all or args.survival:
        print("rendering survival timeline...")
        _write(survival_figure(args.sweep), args.outdir, "survival_timeline.html")
    if args.all or args.heatmap:
        print("rendering death heatmap...")
        _write(death_heatmap(args.sweep), args.outdir, "death_heatmap.html")
    if args.all or args.dispersion:
        print("rendering behavioral dispersion...")
        _write(dispersion_figure(args.sweep, window=args.window), args.outdir,
               "behavioral_dispersion.html")
    if args.all or args.vocabulary:
        print("rendering action vocabulary...")
        _write(vocabulary_figure(args.sweep), args.outdir, "action_distribution.html")
    if args.all or args.network:
        seed_dir = os.path.join(args.sweep, args.seed)
        if not os.path.isdir(seed_dir):
            raise SystemExit(f"Seed dir not found: {seed_dir}")
        print(f"rendering interaction network for {args.seed}...")
        _write(interaction_network(args.sweep, seed_dir, args.endpoint),
               args.outdir, "interaction_network.html")


if __name__ == "__main__":
    main()