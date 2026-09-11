"""Print, for every self-model revision an agent made, the reconstructed
context that plausibly caused it (DESIGN.md section 5.4).

Free by default: deterministic reconstruction from the existing logs
(self_models.jsonl, ticks.jsonl, events.jsonl), zero LLM calls. Pass --narrate
to also ask the local model for a one-sentence causal hypothesis per trace
(costs one call per revision; clearly labeled as the model's own guess, not
ground truth -- hand-verify per DESIGN.md section 7).

Usage:
    python scripts/trace_causes.py --run runs/live01
    python scripts/trace_causes.py --run runs/live01 --agent agent_0
    python scripts/trace_causes.py --run runs/live01 --narrate
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.analysis.causal_trace import build_traces, render_trace, narrate  # noqa: E402
from aiworld.llm_client import LLMClient, config_from_env  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--agent", default=None, help="only this agent_id")
    ap.add_argument("--narrate", action="store_true",
                    help="ask the local model for a one-sentence causal hypothesis per trace")
    args = ap.parse_args()

    traces = build_traces(args.run)
    if args.agent:
        traces = [t for t in traces if t["agent_id"] == args.agent]
    if not traces:
        print(f"No self-model revisions with an actual change found in {args.run}.")
        return

    model_call = None
    if args.narrate:
        client = LLMClient(config_from_env())
        model_call = lambda p: client.complete(p)  # noqa: E731
        print(f"Narrating via '{client.config.model}' ...\n")

    print(f"{len(traces)} self-model change(s) traced in {args.run}:\n")
    for trace in traces:
        print(render_trace(trace))
        if model_call:
            hypothesis = narrate(trace, model_call)
            print(f"  [model's own causal guess -- hand-verify]: {hypothesis}")
        print()


if __name__ == "__main__":
    main()
