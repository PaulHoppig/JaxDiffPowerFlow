"""
Raw JSON loading and schema validation.

This layer uses only plain Python – no JAX arrays.  Its job is to
read the network JSON file, turn it into typed Python dataclasses
(``RawNetwork``), and raise ``ValueError`` on any structural or
semantic violation before anything reaches JAX.

JSON schema overview
--------------------
{
  "meta":  { "name": "<str>" },
  "base":  { "s_mva": <float>, "v_kv": <float> },
  "buses": [
    {
      "id":        <int>,          -- unique, 0-based preferred
      "name":      "<str>",
      "type":      "slack"|"pq"|"pv",
      "p_mw":      <float>,        -- net injection, generator sign
                                   -- (positive = inject into bus)
      "q_mvar":    <float>,        -- same sign convention
      "v_mag_pu":  <float>,        -- magnitude for slack/PV (default 1.0)
      "v_ang_deg": <float>         -- angle for slack reference (default 0.0)
    }, ...
  ],
  "lines": [
    {
      "id":          <int>,
      "name":        "<str>",      -- optional label
      "from_bus":    <int>,        -- references bus "id"
      "to_bus":      <int>,        -- references bus "id"
      "r_pu":        <float>,      -- series resistance [p.u.]
      "x_pu":        <float>,      -- series reactance  [p.u.]
      "b_shunt_pu":  <float>       -- total line charging susceptance [p.u.]
    }, ...
  ]
}
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Raw schema dataclasses  (plain Python, zero JAX)
# ---------------------------------------------------------------------------


@dataclass
class RawBase:
    s_mva: float
    v_kv: float


@dataclass
class RawBus:
    id: int
    name: str
    type: str                    # "slack" | "pq" | "pv"
    p_mw: float = 0.0            # net injection [MW], generator sign
    q_mvar: float = 0.0          # net injection [MVAR], generator sign
    v_mag_pu: float = 1.0        # magnitude setpoint for slack / PV
    v_ang_deg: float = 0.0       # angle reference for slack bus


@dataclass
class RawLine:
    id: int
    from_bus: int
    to_bus: int
    r_pu: float
    x_pu: float
    b_shunt_pu: float = 0.0
    name: str = field(default="")


@dataclass
class RawNetwork:
    name: str
    base: RawBase
    buses: list[RawBus]          # sorted ascending by id after loading
    lines: list[RawLine]         # sorted ascending by id after loading


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_BUS_TYPES = {"slack", "pq", "pv"}


def _dataclass_from_dict(cls, data: dict):
    """Construct a dataclass from a dict, silently ignoring unknown keys."""
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


def _validate(net: RawNetwork) -> None:
    """Raise ``ValueError`` on any structural or semantic error."""
    bus_ids = [b.id for b in net.buses]

    if len(bus_ids) != len(set(bus_ids)):
        raise ValueError("Duplicate bus IDs detected.")

    slack_buses = [b for b in net.buses if b.type == "slack"]
    if len(slack_buses) != 1:
        raise ValueError(
            f"Exactly one slack bus required, found {len(slack_buses)}."
        )

    for bus in net.buses:
        if bus.type not in _VALID_BUS_TYPES:
            raise ValueError(
                f"Bus {bus.id} ({bus.name!r}): unknown type {bus.type!r}. "
                f"Valid types: {sorted(_VALID_BUS_TYPES)}"
            )

    bus_id_set = set(bus_ids)
    line_ids = [ln.id for ln in net.lines]

    if len(line_ids) != len(set(line_ids)):
        raise ValueError("Duplicate line IDs detected.")

    for ln in net.lines:
        if ln.from_bus not in bus_id_set:
            raise ValueError(
                f"Line {ln.id}: from_bus={ln.from_bus} not in bus list."
            )
        if ln.to_bus not in bus_id_set:
            raise ValueError(
                f"Line {ln.id}: to_bus={ln.to_bus} not in bus list."
            )
        if ln.from_bus == ln.to_bus:
            raise ValueError(
                f"Line {ln.id}: self-loop (from_bus == to_bus == {ln.from_bus})."
            )
        if abs(complex(ln.r_pu, ln.x_pu)) < 1e-12:
            raise ValueError(
                f"Line {ln.id}: near-zero impedance |z| < 1e-12 p.u. "
                "(would produce infinite admittance)."
            )

    if net.base.s_mva <= 0:
        raise ValueError(f"base.s_mva must be positive, got {net.base.s_mva}.")
    if net.base.v_kv <= 0:
        raise ValueError(f"base.v_kv must be positive, got {net.base.v_kv}.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_json(path: str | Path) -> RawNetwork:
    """
    Load and validate a network JSON file.

    Parameters
    ----------
    path : str | Path
        Path to the ``.json`` network file.

    Returns
    -------
    RawNetwork
        Validated raw network with buses/lines sorted by ``id``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If a required top-level key is missing.
    ValueError
        On any semantic validation error.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Network file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    base = _dataclass_from_dict(RawBase, data["base"])
    buses = [_dataclass_from_dict(RawBus, b) for b in data["buses"]]
    lines = [_dataclass_from_dict(RawLine, ln) for ln in data["lines"]]

    # Canonical ordering: ascending id; stable index ↔ id mapping downstream.
    buses.sort(key=lambda b: b.id)
    lines.sort(key=lambda ln: ln.id)

    net = RawNetwork(
        name=data.get("meta", {}).get("name", path.stem),
        base=base,
        buses=buses,
        lines=lines,
    )
    _validate(net)
    return net
