"""Tag every logged action with a MACHIAVELLI-style behavior ontology
(DESIGN.md section 7) and print a per-agent breakdown.

Free by default: a deterministic heuristic over the mechanics, zero LLM calls,
zero cost. Pass --llm to fill in whatever the heuristic can't confidently place
("unclassified") using the local model, one batch of actions per call.

Usage:
    python scripts/tag_behaviors.py --run runs/live01
    python scripts/tag_behaviors.py --run runs/live01 --llm
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.analysis.behavior_tags import tag_run, tag_summary, BEHAVIOR_ONTOLOGY  # noqa: E402
from aiworld.llm_client import LLMClient, config_from_env  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--llm", action="store_true",
                    help="fill 'unclassified' tags via the local model (costs calls)")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--sample", type=int, default=5,
                    help="raw tagged actions to print per agent, for hand-verification")
    args = ap.parse_args()

    model_call = None
    if args.llm:
        client = LLMClient(config_from_env())
        model_call = lambda p: client.complete(p)  # noqa: E731
        print(f"Filling unclassified tags via '{client.config.model}' ...")

    tagged = tag_run(args.run, model_call=model_call, batch_size=args.batch_size)
    if not tagged:
        print(f"No ticks.jsonl entries in {args.run}.")
        return

    summary = tag_summary(tagged)
    print(f"\nBehavior breakdown for {args.run} ({len(tagged)} actions tagged):")
    for agent_id in sorted(summary):
        parts = ", ".join(f"{tag}={pct}%" for tag, pct in
                          sorted(summary[agent_id].items(), key=lambda kv: -kv[1]))
        print(f"  {agent_id}: {parts}")

    print(f"\nOntology used: {', '.join(BEHAVIOR_ONTOLOGY)}")
    print("\nDESIGN.md section 7: always hand-verify a sample of automated tags. "
          f"Sample ({args.sample} per agent):")
    by_agent: dict[str, list[dict]] = {}
    for t in tagged:
        by_agent.setdefault(t["agent_id"], []).append(t)
    for agent_id in sorted(by_agent):
        print(f"  {agent_id}:")
        for t in by_agent[agent_id][:args.sample]:
            print(f"    tick {t['tick']}: '{t['action']}' -> {t['tag']}")


if __name__ == "__main__":
    main()
