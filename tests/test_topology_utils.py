"""
Tests for diffpf.io.topology_utils  (Union-Find bus merging).

Covered invariants
------------------
- Isolated buses map to themselves
- Closed bus-bus switch merges two buses into the same representative
- Chain of switches: A-B-C all map to the same representative
- Multiple disjoint groups stay separate
- Unknown switch endpoint raises ValueError
- Filtering of inactive lines (no JAX needed – pure Python helper)
"""

from __future__ import annotations

import pytest

from diffpf.io.topology_utils import find_representative, merge_buses


# ---------------------------------------------------------------------------
# find_representative tests
# ---------------------------------------------------------------------------


def test_find_representative_self():
    parent = {0: 0, 1: 1}
    assert find_representative(parent, 0) == 0
    assert find_representative(parent, 1) == 1


def test_find_representative_chain():
    """0 -> 1 -> 2  (root is 2).  Path compression must fix it."""
    parent = {0: 1, 1: 2, 2: 2}
    rep = find_representative(parent, 0)
    assert rep == 2
    # Path compression: parent[0] should now point directly to 2
    assert parent[0] == 2


# ---------------------------------------------------------------------------
# merge_buses tests
# ---------------------------------------------------------------------------


def test_isolated_buses_map_to_themselves():
    mapping = merge_buses([0, 1, 2], [])
    assert mapping[0] == 0
    assert mapping[1] == 1
    assert mapping[2] == 2


def test_single_closed_switch_merges_two_buses():
    mapping = merge_buses([0, 1, 2, 3], [(0, 1)])
    assert mapping[0] == mapping[1], "0 and 1 should share a representative"
    assert mapping[2] == 2, "2 is not connected"
    assert mapping[3] == 3, "3 is not connected"


def test_chain_of_switches_all_same_representative():
    """0-1-2 closed → all three buses must share one representative."""
    mapping = merge_buses([0, 1, 2], [(0, 1), (1, 2)])
    assert mapping[0] == mapping[1] == mapping[2]


def test_two_disjoint_groups_stay_separate():
    """0-1 and 2-3 are separate groups."""
    mapping = merge_buses([0, 1, 2, 3], [(0, 1), (2, 3)])
    assert mapping[0] == mapping[1]
    assert mapping[2] == mapping[3]
    assert mapping[0] != mapping[2], "groups 0-1 and 2-3 must not merge"


def test_representative_is_smallest_id():
    """The smaller id should become the representative (deterministic)."""
    mapping = merge_buses([5, 10], [(5, 10)])
    assert mapping[5] == 5
    assert mapping[10] == 5


def test_merge_raises_for_unknown_switch_endpoint():
    with pytest.raises(ValueError, match="99"):
        merge_buses([0, 1, 2], [(0, 99)])


def test_merge_buses_non_zero_based_ids():
    """Works with arbitrary (non-sequential) bus IDs."""
    bus_ids = [10, 20, 30]
    mapping = merge_buses(bus_ids, [(10, 20)])
    assert mapping[10] == mapping[20]
    assert mapping[30] == 30


# ---------------------------------------------------------------------------
# Inactive line filter smoke test
# ---------------------------------------------------------------------------


def test_active_line_filter():
    """
    Simple demonstration: inactive lines can be filtered by a boolean flag
    before building the topology (pure Python, no JAX needed here).
    """
    # Lines represented as (from_bus, to_bus, active)
    all_lines = [
        (0, 1, True),
        (1, 2, False),  # inactive
        (0, 2, True),
    ]
    active_lines = [(f, t) for f, t, active in all_lines if active]
    assert active_lines == [(0, 1), (0, 2)]
