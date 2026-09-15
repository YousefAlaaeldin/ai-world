"""Phase 6: Cohen's kappa and the inter-rater usability gate."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.analysis.interrater import (  # noqa: E402
    RatingRecord,
    cohens_kappa,
    kappa_from_rating_sheet,
    load_rating_sheet,
    usable_labels,
)


def test_kappa_is_one_for_perfect_agreement():
    a = ["A", "A", "B", "B"]
    b = ["A", "A", "B", "B"]
    assert cohens_kappa(a, b) == pytest.approx(1.0)


def test_kappa_is_zero_at_chance_level():
    a = ["A", "A", "B", "B"]
    b = ["A", "B", "A", "B"]
    assert cohens_kappa(a, b) == pytest.approx(0.0)


def test_kappa_is_negative_for_systematic_disagreement():
    a = ["A", "A", "B", "B"]
    b = ["B", "B", "A", "A"]
    assert cohens_kappa(a, b) == pytest.approx(-1.0)


def test_kappa_handles_more_than_two_labels():
    a = ["A", "B", "C", "A", "B", "C"]
    b = ["A", "B", "C", "A", "B", "C"]
    assert cohens_kappa(a, b) == pytest.approx(1.0)


def test_kappa_degenerate_single_label_returns_zero_not_nan():
    a = ["A", "A", "A"]
    b = ["A", "A", "A"]
    assert cohens_kappa(a, b) == 0.0


def test_kappa_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same number"):
        cohens_kappa(["A"], ["A", "B"])


def test_kappa_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one item"):
        cohens_kappa([], [])


def test_usable_labels_gate():
    assert usable_labels(0.75) is True
    assert usable_labels(0.65) is False
    assert usable_labels(0.5, threshold=0.4) is True


def test_load_rating_sheet_and_compute_kappa(tmp_path):
    path = tmp_path / "ratings.csv"
    path.write_text(
        "item_id,rater,label\n"
        "t1,alice,pro_social\n"
        "t1,bob,pro_social\n"
        "t2,alice,power_seeking\n"
        "t2,bob,power_seeking\n"
        "t3,alice,neutral\n"
        "t3,bob,pro_social\n",
        encoding="utf-8",
    )
    records = load_rating_sheet(str(path))
    assert len(records) == 6
    assert isinstance(records[0], RatingRecord)

    kappa = kappa_from_rating_sheet(records, "alice", "bob")
    assert -1.0 <= kappa <= 1.0


def test_kappa_from_rating_sheet_only_uses_items_labeled_by_both_raters(tmp_path):
    path = tmp_path / "partial.csv"
    path.write_text(
        "item_id,rater,label\n"
        "t1,alice,pro_social\n"
        "t1,bob,pro_social\n"
        "t2,alice,neutral\n"
        "t2,bob,neutral\n"
        "t3,alice,power_seeking\n",  # bob never rated t3 -- must be dropped
        encoding="utf-8",
    )
    records = load_rating_sheet(str(path))
    kappa = kappa_from_rating_sheet(records, "alice", "bob")
    # t1 and t2 are shared and agree, with two distinct labels -> real
    # (non-degenerate) perfect agreement, unlike a single shared item/label.
    assert kappa == pytest.approx(1.0)


def test_kappa_from_rating_sheet_raises_when_no_items_are_shared(tmp_path):
    path = tmp_path / "disjoint.csv"
    path.write_text(
        "item_id,rater,label\nt1,alice,pro_social\nt2,bob,pro_social\n",
        encoding="utf-8")
    records = load_rating_sheet(str(path))
    with pytest.raises(ValueError, match="no items labeled by both"):
        kappa_from_rating_sheet(records, "alice", "bob")


def test_load_rating_sheet_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("item_id,label\nt1,pro_social\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_rating_sheet(str(path))
