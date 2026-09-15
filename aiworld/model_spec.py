"""ModelSpec: declared provenance for a model used in an experiment run.

Required before a cross-model-family replication (evidence gate #6,
docs/EXPERIMENT_PROTOCOL.md: "replicated with at least one additional pinned
model family before any general statement about LLMs") can be claimed --
"model as an experimental factor" needs the factor actually recorded, not
merely inferred from whatever AIWORLD_MODEL happened to be set to at
runtime.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import yaml

DEFAULT_MODEL_REGISTRY_PATH = "configs/models.example.yaml"


@dataclasses.dataclass
class ModelSpec:
    name: str
    family: str
    size_label: str
    param_count: int | None
    model_identifier: str
    weight_digest: str | None
    tokenizer: str
    quantization: str
    serving_stack: str
    decode_settings: dict[str, Any]
    capability_score_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_model_spec(name: str, path: str = DEFAULT_MODEL_REGISTRY_PATH) -> ModelSpec:
    with open(path, encoding="utf-8") as f:
        registry = yaml.safe_load(f) or {}
    if name not in registry:
        choices = ", ".join(sorted(registry))
        raise ValueError(f"No model spec named {name!r} in {path!r}; "
                         f"available: {choices}.")
    entry = dict(registry[name])
    required = ("family", "size_label", "model_identifier", "tokenizer", "serving_stack")
    missing = [key for key in required if key not in entry]
    if missing:
        raise ValueError(f"model spec {name!r} in {path!r} is missing fields: {missing}")
    return ModelSpec(
        name=name,
        family=entry["family"],
        size_label=entry["size_label"],
        param_count=entry.get("param_count"),
        model_identifier=entry["model_identifier"],
        weight_digest=entry.get("weight_digest"),
        tokenizer=entry["tokenizer"],
        quantization=entry.get("quantization", "unspecified"),
        serving_stack=entry["serving_stack"],
        decode_settings=entry.get("decode_settings") or {},
        capability_score_source=entry.get("capability_score_source"),
    )


def list_model_specs(path: str = DEFAULT_MODEL_REGISTRY_PATH) -> list[str]:
    with open(path, encoding="utf-8") as f:
        registry = yaml.safe_load(f) or {}
    return sorted(registry)
