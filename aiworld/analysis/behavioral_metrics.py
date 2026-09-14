"""Behavioral metrics: measure what agents DO, not what they say.

Tranche 3 (docs/REDESIGN_PLAN.md). The paper's claim -- "emergent behavioral
organization under unassigned objectives in LLM agents" -- makes BEHAVIOR the
primary outcome. That's what lets the de-primed baseline (no self-model fed back,
no directive) be analyzed at all: it produces no self-model trajectory, but it
still produces a stream of actions.

Everything reads ticks.jsonl directly -- no LLM, no network. Features are a FIXED
numeric schema (FEATURE_NAMES), i.e. a stable representation trained/defined once,
NOT a per-tick TF-IDF refit -- so vectors are comparable within a run, across
runs, and across conditions (the property the review flagged as missing).

The features cover action mix, exploration, and sociality, plus the signals most
relevant to "emergent purpose":
  - DISCRETIONARY activity: what an agent does with slack time, when NOT under
    viability pressure. A candidate emergent purpose must be costly/selective --
    something it does when merely staying viable doesn't require it.
  - OVER-ACQUISITION: consuming when its need is already low.
  - The Tranche-4 good/bad action spectrum: gather (store), transfer (give to
    another), build (public good), take (steal), attack (harm/kill). These are
    what let the MACHIAVELLI-style ontology be grounded in real mechanics.

LAYOUT NOTE: FEATURE_NAMES grew with Tranche 4, so a feature vector from a pre-T4
run has fewer columns. Don't cluster across that boundary -- feature_matrix would
produce ragged rows. Analyze runs from one action-space generation together.
"""

from __future__ import annotations

import numpy as np

from .trajectories import load_ticks


NEED_THRESHOLD = 0.5           # a lethal drive >= this = "under viability pressure"
LETHAL_DRIVE_KEYS = ("hunger", "thirst")   # internal log keys (unchanged by vocabulary)

# The fixed behavioral fingerprint. Order matters -- it defines the vector layout.
FEATURE_NAMES = [
    "frac_consume", "frac_move", "frac_talk", "frac_wait",
    "exploration_coverage", "move_rate",
    "talk_rate", "social_partners_frac",
    "discretionary_activity", "over_acquisition", "survival_frac",
    # Tranche 4 good/bad action spectrum:
    "gather_rate", "transfer_rate", "build_rate", "take_rate", "attack_rate",
]

# Action-mix categories used by the over-time divergence metric.
_MIX_KINDS = ("consume", "move", "talk", "wait",
              "gather", "transfer", "build", "take", "attack")


def _under_pressure(rec: dict) -> bool:
    drives = rec.get("drives", {})
    return any(float(drives.get(k, 0.0)) >= NEED_THRESHOLD for k in LETHAL_DRIVE_KEYS)


def _action_kind(action: str) -> str:
    if action in ("consume", "gather", "build"):
        return action
    for prefix in ("move", "talk", "transfer", "take", "attack"):
        if action.startswith(prefix + ":"):
            return prefix
    return "wait"   # 'wait' and any no-op/failed action string


def agent_features(records: list[dict], total_ticks: int, n_agents: int,
                   all_locations: set[str]) -> dict[str, float]:
    """Fixed-schema behavioral feature vector for ONE agent from its tick records."""
    recs = sorted(records, key=lambda r: r["tick"])
    n = len(recs)
    if n == 0:
        return {k: 0.0 for k in FEATURE_NAMES}

    kinds = [_action_kind(r["chosen_action"]) for r in recs]
    def frac(kind: str) -> float:
        return sum(1 for k in kinds if k == kind) / n

    positions = [r["position"] for r in recs]
    visited = set(positions)
    coverage = len(visited) / len(all_locations) if all_locations else 0.0
    if n > 1:
        move_rate = sum(1 for i in range(1, n) if positions[i] != positions[i - 1]) / (n - 1)
    else:
        move_rate = 0.0

    partners = {r["chosen_action"].split(":", 1)[1]
                for r in recs if r["chosen_action"].startswith("talk:")}
    social_partners_frac = len(partners) / (n_agents - 1) if n_agents > 1 else 0.0

    not_pressure = [r for r, k in zip(recs, kinds) if not _under_pressure(r)]
    if not_pressure:
        discretionary = sum(1 for r in not_pressure
                            if _action_kind(r["chosen_action"]) != "wait") / len(not_pressure)
    else:
        discretionary = 0.0

    consume_recs = [r for r, k in zip(recs, kinds) if k == "consume"]
    if consume_recs:
        over_acq = sum(1 for r in consume_recs if not _under_pressure(r)) / len(consume_recs)
    else:
        over_acq = 0.0

    survival_frac = (max(r["tick"] for r in recs) / total_ticks) if total_ticks else 0.0

    return {
        "frac_consume": frac("consume"), "frac_move": frac("move"),
        "frac_talk": frac("talk"), "frac_wait": frac("wait"),
        "exploration_coverage": coverage, "move_rate": move_rate,
        "talk_rate": frac("talk"), "social_partners_frac": social_partners_frac,
        "discretionary_activity": discretionary, "over_acquisition": over_acq,
        "survival_frac": survival_frac,
        "gather_rate": frac("gather"), "transfer_rate": frac("transfer"),
        "build_rate": frac("build"), "take_rate": frac("take"),
        "attack_rate": frac("attack"),
    }


def behavioral_features_run(run_dir: str) -> dict[str, dict[str, float]]:
    """Per-agent behavioral feature vectors for one run. {agent_id: {feature: value}}."""
    ticks = load_ticks(run_dir)
    if not ticks:
        return {}
    total_ticks = max(r["tick"] for r in ticks)
    all_locations = {r["position"] for r in ticks}
    agent_ids = sorted({r["agent_id"] for r in ticks})
    by_agent: dict[str, list[dict]] = {aid: [] for aid in agent_ids}
    for r in ticks:
        by_agent[r["agent_id"]].append(r)
    return {aid: agent_features(recs, total_ticks, len(agent_ids), all_locations)
            for aid, recs in by_agent.items()}


def feature_matrix(features: dict[str, dict[str, float]]) -> tuple[list[str], np.ndarray]:
    """Turn {agent_id: {feature: value}} into (ordered_ids, matrix) with columns in
    FEATURE_NAMES order -- the fixed layout every downstream step relies on."""
    ids = sorted(features)
    mat = np.array([[features[i][f] for f in FEATURE_NAMES] for i in ids], dtype=float)
    return ids, mat


def _standardize(mat: np.ndarray) -> np.ndarray:
    """Z-score each column; zero-variance columns become 0 (guard div-by-zero)."""
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0)
    sd_safe = np.where(sd == 0, 1.0, sd)
    out = (mat - mu) / sd_safe
    out[:, sd == 0] = 0.0
    return out


def behavioral_divergence_over_time(run_dir: str, window: int = 5) -> dict[int, float]:
    """Do agents' BEHAVIORS diverge over time? (The behavioral analogue of
    trajectories.divergence_over_time, which measured self-model text.)

    Splits the run into `window`-tick blocks; in each block, builds every alive
    agent's action-mix vector (consume/move/talk/wait fractions within the block),
    standardizes across agents, and returns {block_end_tick: mean pairwise distance}.
    Rising = behavioral differentiation.
    """
    ticks = load_ticks(run_dir)
    if not ticks:
        return {}
    total = max(r["tick"] for r in ticks)
    out: dict[int, float] = {}
    for end in range(window, total + 1, window):
        start = end - window
        block = [r for r in ticks if start < r["tick"] <= end]
        by_agent: dict[str, list[str]] = {}
        for r in block:
            by_agent.setdefault(r["agent_id"], []).append(_action_kind(r["chosen_action"]))
        vecs = []
        for kinds in by_agent.values():
            n = len(kinds)
            vecs.append([sum(1 for k in kinds if k == kind) / n
                         for kind in _MIX_KINDS])
        if len(vecs) < 2:
            continue
        v = _standardize(np.array(vecs, dtype=float))
        n = len(v)
        dists = [float(np.linalg.norm(v[i] - v[j]))
                 for i in range(n) for j in range(i + 1, n)]
        out[end] = float(np.mean(dists)) if dists else 0.0
    return out


def cluster_behaviors(run_dirs: list[str], k_range=range(2, 7),
                      random_state: int = 0) -> dict:
    """Cluster whole-run agent behavior across many runs, choosing k by silhouette
    (NOT a forced k=5), with a null baseline so recurrence isn't mistaken for
    structure.

    The representation is standardized ONCE across all agent-vectors from every run
    (the fixed-representation requirement). For each k it reports the silhouette,
    and compares the best real silhouette to the silhouette on column-shuffled data
    (which destroys cross-feature structure) -- if real is not clearly above null,
    say so rather than asserting attractors exist.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    all_ids: list[tuple[str, str]] = []
    rows: list[list[float]] = []
    for run_dir in run_dirs:
        feats = behavioral_features_run(run_dir)
        ids, mat = feature_matrix(feats)
        for i, aid in enumerate(ids):
            all_ids.append((run_dir, aid))
            rows.append(list(mat[i]))
    if len(rows) < max(k_range) + 1:
        return {"error": f"only {len(rows)} agent-vectors; need > max k ({max(k_range)})"}

    X = _standardize(np.array(rows, dtype=float))

    def best_silhouette(data: np.ndarray) -> tuple[int, float, dict[int, float]]:
        scores: dict[int, float] = {}
        for k in k_range:
            if k >= len(data):
                continue
            labels = KMeans(n_clusters=k, n_init=10,
                            random_state=random_state).fit_predict(data)
            scores[k] = float(silhouette_score(data, labels))
        best_k = max(scores, key=scores.get)
        return best_k, scores[best_k], scores

    best_k, best_score, score_table = best_silhouette(X)

    rng = np.random.default_rng(random_state)
    shuffled = X.copy()
    for c in range(shuffled.shape[1]):
        rng.shuffle(shuffled[:, c])
    _, null_score, _ = best_silhouette(shuffled)

    labels = KMeans(n_clusters=best_k, n_init=10,
                    random_state=random_state).fit_predict(X)
    clusters: dict[int, list[dict]] = {i: [] for i in range(best_k)}
    for (run_dir, aid), lab in zip(all_ids, labels):
        clusters[int(lab)].append({"run_dir": run_dir, "agent_id": aid})

    # Mean feature profile per cluster, in ORIGINAL units, for hand-labeling.
    raw = np.array(rows, dtype=float)
    profiles: dict[int, dict[str, float]] = {}
    for c in range(best_k):
        members = [i for i, lab in enumerate(labels) if lab == c]
        prof = raw[members].mean(axis=0)
        profiles[c] = {f: round(float(prof[j]), 3) for j, f in enumerate(FEATURE_NAMES)}

    return {
        "n_agent_vectors": len(rows),
        "selected_k": best_k,
        "silhouette": round(best_score, 4),
        "silhouette_by_k": {k: round(v, 4) for k, v in score_table.items()},
        "null_silhouette": round(null_score, 4),
        "clusters_beat_null": best_score > null_score + 0.05,
        "cluster_profiles": profiles,
        "clusters": clusters,
        "note": "Hand-label clusters from their profiles. 'clusters_beat_null' "
                "false => the apparent structure is not above chance; do not claim "
                "attractors. Check recurrence across distinct run_dirs.",
    }
