"""Causal-trace tool: link a self-model delta to the events that plausibly
triggered it (DESIGN.md section 5.4 -- "an agent that became a 'protector'
after an early near-death rescue").

Reconstructs, for every self-model revision an agent made, what happened
between that reflection and the previous one, purely from the existing logs
(self_models.jsonl, ticks.jsonl, events.jsonl) -- no LLM calls needed for the
deterministic trace itself. An optional narrate() layer asks the model for a
one-sentence causal hypothesis over that reconstructed timeline, clearly
labeled as the model's own guess about its own past (DESIGN.md section 7:
hand-verify a sample -- that applies doubly here).
"""

from __future__ import annotations

from typing import Callable

from .trajectories import load_self_models, load_ticks, load_events

_FIELDS = ("who_i_am", "what_matters", "what_i_want_now")


def self_model_diff(prev: dict, curr: dict) -> dict[str, tuple[str, str]]:
    """Which of the three self-model fields changed between two versions, and
    their (old, new) values. Empty dict means the agent's reflection repeated
    itself unchanged -- a real, recordable outcome (self_model.py's revise()
    docstring: "if nothing has changed, repeat them unchanged"), not a bug.
    """
    diff = {}
    for field in _FIELDS:
        old, new = prev.get(field, ""), curr.get(field, "")
        if old != new:
            diff[field] = (old, new)
    return diff


def build_traces(run_dir: str) -> list[dict]:
    """One trace per self-model revision that actually changed something, for
    every agent in the run. Each trace: the diff, the triggering_event tag
    already logged on the new version (includes the "[scheduled]"/"[crisis]"
    reason from fix #6), every action the agent took in the tick window since
    its previous reflection, and every world event logged in that window.
    """
    self_models = load_self_models(run_dir)
    ticks = load_ticks(run_dir)
    events = load_events(run_dir)

    by_agent_versions: dict[str, list[dict]] = {}
    for rec in self_models:
        by_agent_versions.setdefault(rec["agent_id"], []).append(rec)
    for versions in by_agent_versions.values():
        versions.sort(key=lambda r: r.get("version", 0))

    ticks_by_agent: dict[str, list[dict]] = {}
    for r in ticks:
        ticks_by_agent.setdefault(r["agent_id"], []).append(r)
    for v in ticks_by_agent.values():
        v.sort(key=lambda r: r["tick"])

    traces = []
    for agent_id, versions in by_agent_versions.items():
        for prev, curr in zip(versions, versions[1:]):
            diff = self_model_diff(prev, curr)
            if not diff:
                continue
            start_tick = prev.get("revised_at_tick", 0)
            end_tick = curr.get("revised_at_tick", 0)
            window_ticks = [t for t in ticks_by_agent.get(agent_id, [])
                            if start_tick < t["tick"] <= end_tick]
            window_events = [e for e in events
                             if start_tick < e["tick"] <= end_tick]
            traces.append({
                "agent_id": agent_id,
                "from_version": prev.get("version"),
                "to_version": curr.get("version"),
                "tick_window": (start_tick, end_tick),
                "diff": diff,
                "triggering_event": curr.get("triggering_event", ""),
                "actions_in_window": [
                    {"tick": t["tick"], "action": t["chosen_action"]}
                    for t in window_ticks],
                "world_events_in_window": window_events,
            })
    return traces


def render_trace(trace: dict) -> str:
    """Plain-text rendering of one trace, for CLI/log output."""
    start, end = trace["tick_window"]
    lines = [
        f"agent {trace['agent_id']}: v{trace['from_version']} -> v{trace['to_version']} "
        f"(ticks {start}-{end})",
        f"  triggering_event: {trace['triggering_event']}",
    ]
    for field, (old, new) in trace["diff"].items():
        lines.append(f"  {field}: {old!r} -> {new!r}")
    if trace["actions_in_window"]:
        acts = ", ".join(f"{a['tick']}:{a['action']}" for a in trace["actions_in_window"])
        lines.append(f"  actions in window: {acts}")
    else:
        lines.append("  actions in window: (none logged)")
    for e in trace["world_events_in_window"]:
        lines.append(f"  world event @ tick {e['tick']}: {e['kind']} -- {e['detail']}")
    return "\n".join(lines)


def narrate(trace: dict, model_call: Callable[[str], str]) -> str:
    """Ask the model for a ONE-SENTENCE causal hypothesis over a reconstructed
    trace. This is the model's own guess about its own past, not ground truth
    -- DESIGN.md section 7 says hand-verify a sample of automated tags/labels,
    which applies doubly here since the model is narrating ITS OWN causal
    story and has every incentive (from the prompt) to produce a tidy one even
    when the real answer is "unclear."
    """
    diff_text = "\n".join(f"{f}: was {old!r}, became {new!r}"
                          for f, (old, new) in trace["diff"].items())
    actions_text = ", ".join(f"tick {a['tick']}: {a['action']}"
                             for a in trace["actions_in_window"]) or "(none logged)"
    prompt = (
        "An agent's self-description changed as follows:\n"
        f"{diff_text}\n\n"
        f"In the ticks leading up to that change, it did: {actions_text}\n"
        f"The logged triggering event was: {trace['triggering_event']}\n\n"
        "In ONE sentence, state the most plausible causal link between what "
        "happened and how the self-description changed. If there's no clear "
        "link, say so plainly rather than inventing one."
    )
    return model_call(prompt).strip()
