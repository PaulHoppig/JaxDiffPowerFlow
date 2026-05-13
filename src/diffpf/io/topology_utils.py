"""
Topology utilities for bus merging and switch handling.

This module provides pure-Python helpers for bus topology pre-processing
before compilation. No pandapower import, no JAX dependency.

Use case
--------
In networks with bus-bus switches (e.g. pandapower switch elements with
``et='b'`` and ``closed=True``), physically connected buses should be
merged into a single representative bus before the power-flow is set up.
The Union-Find (disjoint-set) algorithm efficiently computes these
equivalence classes.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Union-Find (path-compressed)
# ---------------------------------------------------------------------------


def find_representative(parent: dict[int, int], bus_id: int) -> int:
    """
    Path-compressed Union-Find: return the canonical representative of bus_id.

    Parameters
    ----------
    parent : dict[int, int]
        Mutable mapping from bus_id to its current parent.
        Modified in place by path compression.
    bus_id : int
        The bus whose representative is sought.

    Returns
    -------
    int
        The root (representative) of the equivalence class containing bus_id.
    """
    root = bus_id
    while parent[root] != root:
        root = parent[root]
    # Path compression: point every node directly to the root
    current = bus_id
    while parent[current] != root:
        next_node = parent[current]
        parent[current] = root
        current = next_node
    return root


def merge_buses(
    bus_ids: list[int],
    closed_switches: list[tuple[int, int]],
) -> dict[int, int]:
    """
    Compute a mapping from each bus id to its representative.

    Buses connected by closed bus-bus switches are fused into the same
    equivalence class (Union-Find).  Buses without any switch remain
    their own representative.

    Parameters
    ----------
    bus_ids : list[int]
        All bus ids in the network (arbitrary order, need not be 0-based).
    closed_switches : list[tuple[int, int]]
        Each tuple ``(bus_a, bus_b)`` represents a closed bus-bus switch.
        Both bus ids must appear in ``bus_ids``.

    Returns
    -------
    dict[int, int]
        Mapping ``old_bus_id -> representative_bus_id``.
        For buses not involved in any switch the representative is the
        bus itself (identity mapping).

    Raises
    ------
    ValueError
        If a switch references a bus id not listed in ``bus_ids``.

    Examples
    --------
    >>> mapping = merge_buses([0, 1, 2, 3], [(0, 1), (2, 3)])
    >>> # 0 and 1 share a representative; 2 and 3 share a representative
    >>> assert mapping[0] == mapping[1]
    >>> assert mapping[2] == mapping[3]
    >>> assert mapping[0] != mapping[2]
    """
    bus_id_set = set(bus_ids)

    # Validate switch endpoints
    for a, b in closed_switches:
        if a not in bus_id_set:
            raise ValueError(f"Switch endpoint {a} is not a known bus id.")
        if b not in bus_id_set:
            raise ValueError(f"Switch endpoint {b} is not a known bus id.")

    # Initialise: each bus is its own representative
    parent: dict[int, int] = {bid: bid for bid in bus_ids}

    # Union step
    for a, b in closed_switches:
        ra = find_representative(parent, a)
        rb = find_representative(parent, b)
        if ra != rb:
            # Merge: smaller id becomes the representative (deterministic)
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # Build final flat mapping with path compression applied
    return {bid: find_representative(parent, bid) for bid in bus_ids}
