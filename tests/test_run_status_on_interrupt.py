"""Regression: a Ctrl+C mid-run must not be mislabeled "completed".

KeyboardInterrupt subclasses BaseException, not Exception. Catching only
Exception around world.run(...)/world.step(...) in the run entry points
silently missed it, leaving status="completed" (the pre-try default)
written into the immutable status.json for a run that never reached its
declared tick count. aiworld.analysis.provenance.require_completed_run has
no independent expected-vs-observed tick check, so a truncated run would
have been silently accepted as complete by downstream analysis.

Found while answering "what happens if I stop the pilot run midway?" during
the live Phase 0.5 feasibility pilot (2026-09-15). Fixed identically in
scripts/run_llm.py, scripts/run_stub.py, scripts/run_sampling_null.py, and
scripts/run_branches.py (all four share the same try/except BaseException/
finally pattern); exercised here against run_stub.py since it needs no LLM
client to drive end-to-end.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "scripts")))

import run_stub  # noqa: E402
from aiworld.sim.world import World  # noqa: E402


def test_interrupted_stub_run_is_labeled_aborted_not_completed(tmp_path, monkeypatch):
    real_build_world = run_stub.build_world

    def build_world_in_tmp(cfg, run_id, n_agents, seed, mortality, root="runs"):
        return real_build_world(cfg, run_id, n_agents, seed, mortality,
                                root=str(tmp_path))

    monkeypatch.setattr(run_stub, "build_world", build_world_in_tmp)

    real_step = World.step

    def flaky_step(self, policy, reflect_every=5, reflect_fn=None):
        if self.tick_count >= 2:
            raise KeyboardInterrupt
        return real_step(self, policy, reflect_every=reflect_every,
                         reflect_fn=reflect_fn)

    monkeypatch.setattr(World, "step", flaky_step)
    monkeypatch.setattr(sys, "argv", [
        "run_stub.py",
        "--config", "configs/experiment_deprimed_baseline.yaml",
        "--agents", "2",
        "--ticks", "10",
        "--run-id", "interrupt_test",
    ])

    with pytest.raises(KeyboardInterrupt):
        run_stub.main()

    status_path = tmp_path / "interrupt_test" / "status.json"
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "aborted"
    assert payload["observed_ticks"] < payload["expected_ticks"]
