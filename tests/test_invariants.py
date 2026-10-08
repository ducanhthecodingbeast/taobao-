"""Invariant tests. These are the checks that catch silent label corruption.

Deliberately cheap: they must run in seconds on the sampled artifacts, not rescan 76 M rows.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tmm.config import PATHS
from tmm.io import HistoryStore, SortedKeyIndex
from tmm.metrics import classification, compare, per_user_ranking
from tmm.vocab import ItemVocab


# --------------------------------------------------------------------------------------
# vocabulary / embedding plumbing
# --------------------------------------------------------------------------------------
def test_vocab_is_sorted_and_encodes_round_trip():
    v = ItemVocab.build()
    assert np.all(np.diff(v.ids) >= 0), "vocabulary must be sorted for searchsorted"
    sample = v.ids[np.linspace(0, len(v.ids) - 1, 1000).astype(int)]
    idx = v.encode(sample)
    assert (idx != v.pad_index).all()
    assert np.array_equal(v.decode(idx), sample)


def test_vocab_unknown_id_maps_to_pad():
    v = ItemVocab.build()
    # ids are int64 over the whole range; -2**63 and 2**63-1 cannot be real items
    assert v.encode(np.array([-2**63, 2**63 - 1]))[0] in (v.pad_index, 0)
    assert v.pad_index == len(v.ids)


def test_sorted_key_index_finds_and_misses():
    idx = SortedKeyIndex.load(PATHS.emb_keys)
    present = np.asarray(idx.keys[:5])
    rows = idx.find(present)
    assert (rows >= 0).all()
    # a value strictly below the minimum cannot exist
    assert idx.find(np.array([idx.keys[0] - 1]))[0] == -1


def test_embedding_scale_is_one_over_127():
    """The single most consequential constant in the project."""
    from tmm.io import EmbeddingStore

    store = EmbeddingStore.open()
    st = store.quant_stats()
    assert st["scale"] == pytest.approx(1 / 127.0)
    assert store.row_norm_check() == pytest.approx(1.0, abs=0.05)


# --------------------------------------------------------------------------------------
# dataset invariants
# --------------------------------------------------------------------------------------
def test_label_balance_matches_measured():
    p = json.loads((PATHS.stats / "profile.json").read_text())
    for split in ("train", "test"):
        assert p["labels"][split]["pos_rate"] == pytest.approx(0.137, abs=0.002)


def test_candidate_sets_are_complete_and_consistent():
    """If histories and candidates were sourced from the same partial scan, rows_per_user
    would collapse towards ~1 and every ranking metric would be meaningless."""
    p = json.loads((PATHS.stats / "prepare.json").read_text())
    for split in ("train", "test"):
        rpu = p["splits"][split]["rows_per_user"]
        assert 9.0 < rpu < 13.0, f"{split} rows_per_user={rpu} looks like a partial scan"


def test_history_store_alignment():
    hs = HistoryStore.load(PATHS.data / "hist")
    n_users = np.load(PATHS.data / "user_ids.npy").shape[0]
    assert len(hs) == n_users
    b = hs.batch(np.arange(min(8, len(hs))), length=1000, pad=-1)
    assert b.shape == (min(8, len(hs)), 1000)
    assert (np.diff(hs.offsets) > 0).all(), "every kept user must have a non-empty history"


def test_no_inner_join_label_bias():
    b = json.loads((PATHS.bench / "join_duckdb.json").read_text())
    assert b["no_join_loss"]["join_retention"] > 0.999
    assert b["no_join_loss"]["pos_rate_shift"] < 1e-4


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def test_per_user_ranking_is_macro_averaged():
    # user 0: positive ranked 1st of 5; user 1: positive ranked 4th of 5
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.int8)
    user = np.array([0] * 5 + [1] * 5)
    scores = np.array([9, 8, 7, 6, 5, 10, 9, 8, 7, 6], dtype=np.float64)
    r = per_user_ranking(y, scores, user, ks=(1, 5))
    assert r["n_users"] == 2
    assert r["HR@1"] == pytest.approx(0.5)      # only user 0
    assert r["HR@5"] == pytest.approx(1.0)      # both
    assert r["MRR"] == pytest.approx((1.0 + 0.25) / 2)


def test_ndcg_of_a_perfect_ranking_is_one_with_several_positives():
    y = np.array([1, 1, 0, 0], dtype=np.int8)
    r = per_user_ranking(y, np.array([4.0, 3.0, 2.0, 1.0]), np.zeros(4, dtype=np.int64),
                         ks=(1, 3))
    assert r["NDCG@1"] == pytest.approx(1.0)
    assert r["NDCG@3"] == pytest.approx(1.0)


def test_ties_are_not_broken_by_row_order():
    # every user's positive is stored first; a constant scorer must not get HR@1 = 1
    n_users = 3000
    y = np.tile(np.array([1, 0, 0], dtype=np.int8), n_users)
    user = np.repeat(np.arange(n_users), 3)
    r = per_user_ranking(y, np.zeros(len(y)), user, ks=(1,))
    assert r["HR@1"] == pytest.approx(1 / 3, abs=0.03)


def test_users_without_a_positive_are_not_scored():
    y = np.array([1, 0, 0, 0], dtype=np.int8)
    user = np.array([0, 0, 1, 1])
    r = per_user_ranking(y, np.array([2.0, 1.0, 2.0, 1.0]), user, ks=(1,))
    assert r["n_users"] == 1 and r["n_users_total"] == 2
    assert r["HR@1"] == pytest.approx(1.0)


def test_classification_auc_perfect_and_random():
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    assert classification(y, np.array([-2.0, -1.0, 1.0, 2.0]))["AUC"] == pytest.approx(1.0)
    assert classification(y, np.array([-2.0, -1.0, 1.0, 2.0])[::-1])["AUC"] == pytest.approx(0.0)


def test_compare_refuses_mixed_candidate_sizes():
    """Rendle (arXiv:1912.02263): sampled metrics do not preserve model order."""
    a = {"AUC": 0.8, "candidate_set_size": 11}
    b = {"AUC": 0.9, "candidate_set_size": 500}
    assert compare({"a": a, "b": b}, "AUC")["comparable"] is False
    assert compare({"a": a, "b": dict(a, AUC=0.9)}, "AUC")["comparable"] is True

