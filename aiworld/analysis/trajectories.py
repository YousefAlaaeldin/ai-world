"""Analysis: turn logged self-model versions into the experiment's actual results.

Two questions this answers (DESIGN.md §5):
  1. DIVERGENCE — do agents that started identical drift apart? (population variance
     of self-model embeddings vs. tick)
  2. TAXONOMY — do stabilized self-models cluster into recurring 'purpose attractors'
     across independent runs?

Embeddings are pluggable. Provide an `embed_fn(list[str]) -> np.ndarray`. For a
quick offline pass you can use a local sentence-transformer; for consistency with the
sim's model you can use the same provider. This module never hard-codes a provider.

Requires: numpy, scikit-learn (see requirements.txt). No network needed if you pass
a local embed_fn.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Callable

import numpy as np


def tfidf_embed_fn(texts: list[str]) -> np.ndarray:
    """A local, zero-cost, zero-network embed_fn for a quick offline pass --
    scikit-learn is already a required dependency, nothing extra to install or
    call out to. Bag-of-words/TF-IDF isn't a semantic embedding (it won't catch
    paraphrases: "a gatherer" and "one who collects" look unrelated to it), so
    it's a legitimate first-pass distance metric, not a final one. Swap in a
    real embedding model via the `embed_fn` parameter on divergence_over_time /
    cluster_final_purposes once paraphrase-level nuance actually matters.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer()
    mat = vec.fit_transform(texts)
    return np.asarray(mat.todense())


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


def load_self_models(run_dir: str) -> list[dict]:
    """Load all self-model versions logged in a run."""
    return _load_jsonl(os.path.join(run_dir, "self_models.jsonl"))


def load_ticks(run_dir: str) -> list[dict]:
    """Load all per-tick records logged in a run (aiworld.logging_schema.TickRecord).
    Used by aiworld.analysis.behavior_tags and .causal_trace -- kept here so both
    share one loader instead of duplicating the read loop.
    """
    return _load_jsonl(os.path.join(run_dir, "ticks.jsonl"))


def load_events(run_dir: str) -> list[dict]:
    """Load all world events logged in a run (aiworld.logging_schema.EventRecord)."""
    return _load_jsonl(os.path.join(run_dir, "events.jsonl"))


def _embedding_text(rec: dict) -> str:
    return f"{rec.get('who_i_am','')} || {rec.get('what_matters','')} || " \
           f"{rec.get('what_i_want_now','')}"


def divergence_over_time(
    run_dir: str,
    embed_fn: Callable[[list[str]], np.ndarray],
) -> dict[int, float]:
    """Mean pairwise embedding distance between agents' *current* self-models at each
    revision tick. Rising = differentiation; falling = convergence to monoculture.

    Returns {tick: mean_pairwise_distance}.
    """
    records = load_self_models(run_dir)
    if not records:
        return {}

    # Group latest self-model per agent as of each revision tick.
    by_tick: dict[int, dict[str, str]] = {}
    running: dict[str, str] = {}
    for rec in sorted(records, key=lambda r: r.get("revised_at_tick", 0)):
        tick = rec.get("revised_at_tick", 0)
        running[rec["agent_id"]] = _embedding_text(rec)
        by_tick[tick] = dict(running)  # snapshot

    out: dict[int, float] = {}
    for tick, agent_texts in by_tick.items():
        texts = list(agent_texts.values())
        if len(texts) < 2:
            continue
        vecs = embed_fn(texts)
        vecs = _l2_normalize(vecs)
        # mean pairwise cosine distance
        sims = vecs @ vecs.T
        n = len(texts)
        iu = np.triu_indices(n, k=1)
        mean_dist = float(np.mean(1.0 - sims[iu]))
        out[tick] = mean_dist
    return out


def cluster_final_purposes(
    run_dirs: list[str],
    embed_fn: Callable[[list[str]], np.ndarray],
    n_clusters: int = 5,
) -> dict:
    """Cluster the FINAL self-model of every agent across many runs.

    Recurrent clusters across independent runs = candidate 'purpose attractors',
    the headline result. Returns cluster assignments + the member texts so you can
    hand-label them (hoarder / caretaker / wanderer / ...).
    """
    from sklearn.cluster import KMeans

    finals: list[str] = []
    meta: list[tuple[str, str]] = []  # (run_id, agent_id)
    for run_dir in run_dirs:
        records = load_self_models(run_dir)
        latest: dict[str, dict] = {}
        for rec in records:
            aid = rec["agent_id"]
            if aid not in latest or rec.get("version", 0) > latest[aid].get("version", 0):
                latest[aid] = rec
        for aid, rec in latest.items():
            finals.append(_embedding_text(rec))
            meta.append((os.path.basename(run_dir), aid))

    if len(finals) < n_clusters:
        return {"error": f"only {len(finals)} final self-models; "
                         f"need >= n_clusters ({n_clusters})"}

    vecs = _l2_normalize(embed_fn(finals))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = km.fit_predict(vecs)

    clusters: dict[int, list[dict]] = {i: [] for i in range(n_clusters)}
    for (run_id, aid), text, lab in zip(meta, finals, labels):
        clusters[int(lab)].append(
            {"run_id": run_id, "agent_id": aid, "text": text})
    return {
        "n_final_models": len(finals),
        "n_clusters": n_clusters,
        "clusters": clusters,
        "note": "Hand-label each cluster. Check whether clusters recur across "
                "distinct run_ids — that recurrence is the key finding.",
    }


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def all_run_dirs(root: str = "runs") -> list[str]:
    return sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
