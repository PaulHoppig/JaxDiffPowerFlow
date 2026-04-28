"""
Core data types for the differentiable AC power-flow kernel.

Design contract
---------------
* All numeric leaves are JAX arrays (float64 or int32).
* Frozen dataclasses registered as pytrees via ``register_dataclass``.
* ``meta_fields`` are treated as *static* by JAX (not differentiated,
  not traced); ``data_fields`` are differentiated.
* ``core/`` never imports from ``io/`` or any dict/parsing logic.

Type hierarchy
--------------
Human-friendly input layer (no JAX required):
  BusSpec, LineSpec, NetworkSpec

Compiled, JAX-ready layer (all arrays):
  CompiledTopology  - static topology (indices, counts); meta-only w.r.t. jit
  NetworkParams     - differentiable physical parameters
  PFState           - rectangular voltage state for non-slack buses
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp
from jax import tree_util

# ---------------------------------------------------------------------------
# Human-friendly input specs  (plain Python, no JAX)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BusSpec:
    """Human-readable bus description used before compilation."""

    name: str
    is_slack: bool = False


@dataclass(frozen=True)
class LineSpec:
    """Human-readable Pi-model line description in per-unit."""

    from_bus: int
    to_bus: int
    r_pu: float
    x_pu: float
    b_shunt_pu: float = 0.0


@dataclass(frozen=True)
class TrafoSpec:
    """Human-readable 2-winding transformer in per-unit (Pi-model, system base)."""

    hv_bus: int
    lv_bus: int
    r_pu: float
    x_pu: float
    g_mag_pu: float = 0.0   # magnetising conductance (iron losses)
    b_mag_pu: float = 0.0   # magnetising susceptance (no-load current)
    tap_ratio: float = 1.0  # final tap factor (dimensionless)
    shift_rad: float = 0.0  # phase shift in radians
    name: str = ""


@dataclass(frozen=True)
class ShuntSpec:
    """Human-readable shunt admittance in per-unit."""

    bus: int
    g_pu: float = 0.0   # conductance  (losses)
    b_pu: float = 0.0   # susceptance  (positive = capacitive in generator sign)
    name: str = ""


@dataclass(frozen=True)
class NetworkSpec:
    """
    Human-readable network input: static topology + default parameters.

    Sign convention: p_spec_pu / q_spec_pu use *generator* sign
    (positive = power injected into the bus).
    """

    buses: tuple[BusSpec, ...]
    lines: tuple[LineSpec, ...]
    p_spec_pu: tuple[float, ...]
    q_spec_pu: tuple[float, ...]
    slack_vr_pu: float = 1.0
    slack_vi_pu: float = 0.0
    trafos: tuple[TrafoSpec, ...] = ()
    shunts: tuple[ShuntSpec, ...] = ()
    # PV-bus setpoints: dict mapping bus index → v_set_pu (may be empty)
    v_set_pu: tuple[tuple[int, float], ...] = ()


# ---------------------------------------------------------------------------
# Compiled, JAX-ready layer
# ---------------------------------------------------------------------------


@partial(
    tree_util.register_dataclass,
    data_fields=["from_bus", "to_bus", "variable_buses"],
    meta_fields=["n_bus", "slack_bus"],
)
@dataclass(frozen=True)
class CompiledTopology:
    """
    Static array-based network topology.

    ``meta_fields`` are compile-time constants for jit/grad.
    ``data_fields`` are integer index arrays; treat as
    stop_gradient when differentiating physical parameters.
    """

    n_bus: int
    slack_bus: int
    from_bus: jnp.ndarray     # int32, shape (n_line,)
    to_bus: jnp.ndarray       # int32, shape (n_line,)
    variable_buses: jnp.ndarray  # int32, shape (n_bus - 1,)


@partial(
    tree_util.register_dataclass,
    data_fields=[
        "p_spec_pu",
        "q_spec_pu",
        "g_series_pu",
        "b_series_pu",
        "b_shunt_pu",
        "slack_vr_pu",
        "slack_vi_pu",
        # transformer arrays (empty if no trafos)
        "trafo_g_series_pu",
        "trafo_b_series_pu",
        "trafo_g_mag_pu",
        "trafo_b_mag_pu",
        "trafo_tap_ratio",
        "trafo_shift_rad",
        # shunt arrays (empty if no shunts)
        "shunt_g_pu",
        "shunt_b_pu",
    ],
    meta_fields=[
        "trafo_hv_bus",
        "trafo_lv_bus",
        "shunt_bus",
    ],
)
@dataclass(frozen=True)
class NetworkParams:
    """
    Differentiable per-unit network parameters.

    All fields are JAX float64 arrays (or scalars promoted to 0-d arrays).
    Shapes:
      p_spec_pu, q_spec_pu : (n_bus,)
      g_series_pu, b_series_pu, b_shunt_pu : (n_line,)
      slack_vr_pu, slack_vi_pu : scalar
      trafo_* : (n_trafo,)
      shunt_* : (n_shunt,)
    """

    p_spec_pu: jnp.ndarray
    q_spec_pu: jnp.ndarray
    g_series_pu: jnp.ndarray
    b_series_pu: jnp.ndarray
    b_shunt_pu: jnp.ndarray
    slack_vr_pu: jnp.ndarray
    slack_vi_pu: jnp.ndarray
    # transformer arrays (empty by default → no trafos)
    trafo_g_series_pu: jnp.ndarray = None   # shape (n_trafo,)
    trafo_b_series_pu: jnp.ndarray = None   # shape (n_trafo,)
    trafo_g_mag_pu: jnp.ndarray = None      # shape (n_trafo,)
    trafo_b_mag_pu: jnp.ndarray = None      # shape (n_trafo,)
    trafo_tap_ratio: jnp.ndarray = None     # shape (n_trafo,)
    trafo_shift_rad: jnp.ndarray = None     # shape (n_trafo,)
    trafo_hv_bus: tuple[int, ...] = ()      # meta (static)
    trafo_lv_bus: tuple[int, ...] = ()      # meta (static)
    # shunt arrays (empty by default → no shunts)
    shunt_g_pu: jnp.ndarray = None          # shape (n_shunt,)
    shunt_b_pu: jnp.ndarray = None          # shape (n_shunt,)
    shunt_bus: tuple[int, ...] = ()         # meta (static)


@partial(
    tree_util.register_dataclass,
    data_fields=["vr_pu", "vi_pu"],
    meta_fields=[],
)
@dataclass(frozen=True)
class PFState:
    """
    Rectangular voltage state for the *non-slack* buses.

    V_i = vr_pu[i] + j * vi_pu[i]  (per unit)
    Shape: (n_bus - 1,) each.
    """

    vr_pu: jnp.ndarray
    vi_pu: jnp.ndarray

    def as_vector(self) -> jnp.ndarray:
        """Flatten state into a single real vector [vr | vi]."""
        return jnp.concatenate([self.vr_pu, self.vi_pu], axis=0)

    @classmethod
    def from_vector(cls, vector: jnp.ndarray, n_var: int) -> "PFState":
        """Reconstruct state from a flat real vector."""
        return cls(vr_pu=vector[:n_var], vi_pu=vector[n_var:])
