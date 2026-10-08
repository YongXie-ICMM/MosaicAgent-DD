"""Row-offset votes of registration segments: the stage's numbering wins unless the evidence is real."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import register as R  # noqa: E402


def test_single_weak_edge_cannot_move_a_segment():
    # the 2026-10-07 chip-edge case: one edge at 0.27 for offset -2, noise elsewhere
    acc = {-3: [0.11], -2: [0.27], -1: [0.09], 0: [0.14], 1: [0.08], 2: [0.12], 3: [0.10]}
    off, n_good, _ = R.decide_segment_offset(acc)
    assert n_good == 0, "a single barely-passing edge is no evidence: the caller must inherit"


def test_single_strong_edge_is_enough():
    acc = {-1: [0.82], 0: [0.14], 1: [0.09]}
    assert R.decide_segment_offset(acc)[:2] == (-1, 1)


def test_two_good_edges_beat_one_at_zero():
    acc = {1: [0.42, 0.45], 0: [0.41], -1: [0.1]}
    assert R.decide_segment_offset(acc)[:2] == (1, 2)


def test_equal_evidence_keeps_the_chain_hypothesis():
    acc = {1: [0.44, 0.40], 0: [0.41, 0.39], -1: [0.1]}
    off, n_good, _ = R.decide_segment_offset(acc)
    assert off == 0 and n_good == 2


def test_clear_margin_moves_even_with_equal_counts():
    acc = {1: [0.70, 0.68], 0: [0.30, 0.28]}
    assert R.decide_segment_offset(acc)[0] == 1


def test_no_votes_at_all():
    assert R.decide_segment_offset({})[:2] == (0, 0)
