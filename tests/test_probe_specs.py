"""Phase 2: MapSpec / ProbeSpec loading, validation, and predicate evaluation."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.experiments.specs import (  # noqa: E402
    MapSpec,
    ProbeSpec,
    eval_predicate,
    load_map_spec,
    load_probe_spec,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---- MapSpec ----

def test_load_real_map_specs_across_all_three_splits():
    dev = load_map_spec(os.path.join(REPO_ROOT, "configs/maps/development/dev_hub5.yaml"))
    heldout = load_map_spec(os.path.join(REPO_ROOT, "configs/maps/heldout/heldout_hub6.yaml"))
    confirmation = load_map_spec(
        os.path.join(REPO_ROOT, "configs/maps/confirmation/confirmation_ring5.yaml"))

    assert dev.split == "development"
    assert heldout.split == "heldout"
    assert confirmation.split == "confirmation"
    # Genuinely distinct topologies, not relabelings of one another.
    assert len({dev.content_hash, heldout.content_hash, confirmation.content_hash}) == 3
    assert set(dev.locations) != set(heldout.locations) != set(confirmation.locations)


def test_map_spec_hash_is_stable_across_reloads():
    path = os.path.join(REPO_ROOT, "configs/maps/development/dev_hub5.yaml")
    assert load_map_spec(path).content_hash == load_map_spec(path).content_hash


def test_map_spec_rejects_unknown_split(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: bad\nsplit: staging\nworld:\n  locations: [a]\n  edges: []\n  resources: []\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="split"):
        load_map_spec(str(bad))


def test_map_spec_validate_rejects_edge_to_unknown_location():
    spec = MapSpec(
        name="bad", split="development", locations=["a", "b"],
        edges=[["a", "c"]], resources=[], start_mode="balanced",
        label_mode="natural", action_api_version="v1", content_hash="x")
    with pytest.raises(ValueError, match="unknown location"):
        spec.validate()


def test_map_spec_validate_rejects_resource_at_unknown_location():
    spec = MapSpec(
        name="bad", split="development", locations=["a", "b"],
        edges=[["a", "b"]], resources=[{"location": "c", "drive": "hunger", "quantity": 1}],
        start_mode="balanced", label_mode="natural", action_api_version="v1",
        content_hash="x")
    with pytest.raises(ValueError, match="unknown location"):
        spec.validate()


def test_map_spec_to_world_config_matches_source():
    spec = load_map_spec(os.path.join(REPO_ROOT, "configs/maps/development/dev_hub5.yaml"))
    world_cfg = spec.to_world_config()
    assert world_cfg["locations"] == spec.locations
    assert world_cfg["resources"] == spec.resources


# ---- ProbeSpec ----

_ACTIVE_PROBES = ("probe_01_retain_vs_transfer", "probe_03_harm_with_alternative",
                  "probe_06_ledger_ablation")
_DEFERRED_PROBES = ("probe_02_private_vs_public", "probe_04_blocked_route",
                    "probe_05_cost_ladder", "probe_07_mirror_relabel")


@pytest.mark.parametrize("probe_name", _ACTIVE_PROBES)
def test_active_probes_load_with_status_active(probe_name):
    spec = load_probe_spec(os.path.join(REPO_ROOT, f"configs/probes/{probe_name}.yaml"))
    assert isinstance(spec, ProbeSpec)
    assert spec.status == "active"
    assert spec.id == probe_name
    assert spec.branch_mode in ("microprobe", "ecological")


@pytest.mark.parametrize("probe_name", _DEFERRED_PROBES)
def test_deferred_probes_load_with_status_deferred(probe_name):
    spec = load_probe_spec(os.path.join(REPO_ROOT, f"configs/probes/{probe_name}.yaml"))
    assert spec.status == "deferred"


def test_probe_spec_rejects_unknown_branch_mode(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "id: bad\nconstruct: x\nbranch_mode: chaotic\ncandidate_action: wait\n"
        "required_viable_alternative: wait\n", encoding="utf-8")
    with pytest.raises(ValueError, match="branch_mode"):
        load_probe_spec(str(bad))


def test_probe_spec_rejects_unknown_status(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "id: bad\nconstruct: x\nstatus: maybe\ncandidate_action: wait\n"
        "required_viable_alternative: wait\n", encoding="utf-8")
    with pytest.raises(ValueError, match="status"):
        load_probe_spec(str(bad))


def test_probe_spec_hash_is_stable_across_reloads():
    path = os.path.join(REPO_ROOT, "configs/probes/probe_01_retain_vs_transfer.yaml")
    assert load_probe_spec(path).content_hash == load_probe_spec(path).content_hash


# ---- eval_predicate ----

def test_eval_predicate_ands_every_clause():
    predicate = {
        "pre_action_opportunities.can_transfer": ["truthy", True],
        "pre_action_drives.hunger": ["lt", 0.5],
    }
    assert eval_predicate(predicate, {
        "pre_action_opportunities": {"can_transfer": True},
        "pre_action_drives": {"hunger": 0.1},
    }) is True
    assert eval_predicate(predicate, {
        "pre_action_opportunities": {"can_transfer": True},
        "pre_action_drives": {"hunger": 0.9},  # fails the second clause
    }) is False


def test_eval_predicate_missing_field_resolves_to_not_eligible():
    predicate = {"pre_action_opportunities.can_attack": ["truthy", True]}
    assert eval_predicate(predicate, {}) is False


def test_eval_predicate_comparison_ops():
    state = {"x": 5}
    assert eval_predicate({"x": ["eq", 5]}, state) is True
    assert eval_predicate({"x": ["ne", 5]}, state) is False
    assert eval_predicate({"x": ["gte", 5]}, state) is True
    assert eval_predicate({"x": ["gt", 5]}, state) is False
    assert eval_predicate({"x": ["in", [1, 5, 9]]}, state) is True
    assert eval_predicate({"x": ["not_in", [1, 5, 9]]}, state) is False


def test_eval_predicate_rejects_unknown_op():
    with pytest.raises(ValueError, match="Unknown predicate op"):
        eval_predicate({"x": ["nonsense", 1]}, {"x": 1})


def test_the_probe_01_predicate_actually_matches_pre_action_opportunities_shape():
    """Regression guard: the probe YAML's field paths must match the REAL
    shape World._pre_action_opportunities() logs into TickRecord, or the
    predicate would silently never fire."""
    spec = load_probe_spec(
        os.path.join(REPO_ROOT, "configs/probes/probe_01_retain_vs_transfer.yaml"))
    decision_state = {
        "pre_action_opportunities": {"can_transfer": True},
        "pre_action_drives": {"hunger": 0.1, "thirst": 0.1},
    }
    assert eval_predicate(spec.eligible_state_predicate, decision_state) is True
