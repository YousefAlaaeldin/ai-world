"""Inter-rater reliability for human-coded behavior/moral labels.

docs/EXPERIMENT_PROTOCOL.md requires "blinded human coding plus inter-rater
agreement" before behavioral/moral labels support a result -- nothing in the
codebase implemented that check before this module (this build's Phase 6).
"""

from __future__ import annotations

import csv
import dataclasses
from collections import Counter
from typing import Sequence


def cohens_kappa(rater_a: Sequence[str], rater_b: Sequence[str],
                 labels: Sequence[str] | None = None) -> float:
    """Cohen's kappa for two raters' labels over the SAME ordered items.

    ``kappa = (p_o - p_e) / (1 - p_e)``, where ``p_o`` is observed agreement
    and ``p_e`` is the agreement expected by chance from each rater's own
    marginal label distribution. 1.0 is perfect agreement, 0.0 is
    chance-level, negative values mean worse-than-chance (systematic
    disagreement).
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must label the same number of items")
    n = len(rater_a)
    if n == 0:
        raise ValueError("cohens_kappa requires at least one item")

    if labels is None:
        labels = sorted(set(rater_a) | set(rater_b))

    observed_agreement = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n

    count_a = Counter(rater_a)
    count_b = Counter(rater_b)
    chance_agreement = sum(
        (count_a.get(label, 0) / n) * (count_b.get(label, 0) / n)
        for label in labels
    )

    if chance_agreement >= 1.0:
        # No variance in either rater's marginal distribution to explain --
        # degenerate case (e.g. both raters used exactly one label for
        # everything). Treat as no measurable agreement beyond chance rather
        # than dividing by zero.
        return 0.0
    return (observed_agreement - chance_agreement) / (1 - chance_agreement)


@dataclasses.dataclass
class RatingRecord:
    item_id: str
    rater: str
    label: str


def load_rating_sheet(path: str) -> list[RatingRecord]:
    """Load a simple CSV rating sheet with columns: item_id, rater, label."""
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"item_id", "rater", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"rating sheet {path!r} is missing columns: {sorted(missing)}")
        for row in reader:
            records.append(RatingRecord(item_id=row["item_id"], rater=row["rater"],
                                        label=row["label"]))
    return records


def kappa_from_rating_sheet(records: list[RatingRecord], rater_a: str, rater_b: str) -> float:
    """Cohen's kappa between two named raters from a loaded rating sheet.

    Items must be labeled by BOTH raters; an item labeled by only one is
    dropped from the comparison rather than silently padded with a guess.
    """
    by_item_a = {r.item_id: r.label for r in records if r.rater == rater_a}
    by_item_b = {r.item_id: r.label for r in records if r.rater == rater_b}
    shared_items = sorted(set(by_item_a) & set(by_item_b))
    if not shared_items:
        raise ValueError(f"no items labeled by both {rater_a!r} and {rater_b!r}")
    a_labels = [by_item_a[item] for item in shared_items]
    b_labels = [by_item_b[item] for item in shared_items]
    return cohens_kappa(a_labels, b_labels)


# Landis & Koch (1977)'s commonly cited "substantial agreement" cutoff.
DEFAULT_KAPPA_THRESHOLD = 0.7


def usable_labels(kappa: float, threshold: float = DEFAULT_KAPPA_THRESHOLD) -> bool:
    """Whether inter-rater agreement clears the bar to treat labels as usable
    evidence (docs/EXPERIMENT_PROTOCOL.md's blinded-coding-plus-agreement
    requirement). Below threshold: report the disagreement and do not
    silently proceed with one rater's labels alone.
    """
    return kappa >= threshold
