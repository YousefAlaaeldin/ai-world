"""Phase 8: ModelSpec loading, and run_llm.py's optional --model-spec wiring."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.model_spec import ModelSpec, list_model_specs, load_model_spec  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REGISTRY_PATH = os.path.join(REPO_ROOT, "configs", "models.example.yaml")


def test_list_model_specs_includes_both_pinned_entries():
    names = list_model_specs(REGISTRY_PATH)
    assert "mistral_small_abliterated" in names
    assert "qwen2_5_14b_instruct" in names


def test_load_default_model_spec():
    spec = load_model_spec("mistral_small_abliterated", path=REGISTRY_PATH)
    assert isinstance(spec, ModelSpec)
    assert spec.family == "mistral"
    assert spec.serving_stack == "ollama"
    assert spec.model_identifier == "huihui_ai/mistral-small-abliterated:24b"


def test_load_pinned_second_family_model_spec():
    """The pinned second family (evidence gate #6): a different pretraining
    lineage, served through the same local endpoint -- no new client code."""
    spec = load_model_spec("qwen2_5_14b_instruct", path=REGISTRY_PATH)
    assert spec.family == "qwen2.5"
    assert spec.serving_stack == "ollama"
    assert spec.model_identifier == "qwen2.5:14b-instruct"
    # Different family from the default, confirming this is a real second
    # pretraining/RLHF lineage, not a relabeled copy of the same one.
    default_spec = load_model_spec("mistral_small_abliterated", path=REGISTRY_PATH)
    assert spec.family != default_spec.family


def test_load_model_spec_rejects_unknown_name():
    with pytest.raises(ValueError, match="No model spec named"):
        load_model_spec("nonexistent_model", path=REGISTRY_PATH)


def test_model_spec_to_dict_is_json_safe(tmp_path):
    import json
    spec = load_model_spec("mistral_small_abliterated", path=REGISTRY_PATH)
    json.dumps(spec.to_dict())  # must not raise


def test_load_model_spec_rejects_registry_entry_missing_required_fields(tmp_path):
    path = tmp_path / "bad_registry.yaml"
    path.write_text("incomplete_model:\n  family: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        load_model_spec("incomplete_model", path=str(path))
