"""
Y-bus matrix assembly via the stamping method.

Each Pi-model line contributes four stamps to Y_bus:
  Y[i,i] += y_series + y_shunt/2
  Y[j,j] += y_series + y_shunt/2
  Y[i,j] -= y_series
  Y[j,i] -= y_series

where y_series = 1 / (r + jx)  and  y_shunt = j * b_shunt.

For transformers with tap ratio ``a`` and phase shift ``phi``, the
admittance ``y_t = g + jb`` (series) and shunt admittance ``y_m = g_m + jb_m``
yield the following Pi-model stamps (off-nominal tap, HV=from, LV=to):

  Y[hv, hv] += (y_t + y_m) / a²
  Y[lv, lv] += (y_t + y_m)
  Y[hv, lv] += -y_t / (a · exp(-j·phi)) = -y_t / conj(t)
  Y[lv, hv] += -y_t / (a · exp( j·phi)) = -y_t / t

where t = a · exp(j·phi).

Shunts add directly to the diagonal:
  Y[bus, bus] += g_sh + j·b_sh

The function is pure and JIT-able; topology indices are treated as
compile-time constants by JAX (they live in CompiledTopology.meta_fields
or are accessed via static index).
"""

from __future__ import annotations

import jax.numpy as jnp

from diffpf.core.types import CompiledTopology, NetworkParams


def build_ybus(
    topology: CompiledTopology,
    params: NetworkParams,
) -> jnp.ndarray:
    """
    Assemble the complex n×n bus admittance matrix.

    Parameters
    ----------
    topology : CompiledTopology
        Static network topology (bus/line counts, index arrays).
    params : NetworkParams
        Differentiable line parameters (g_series, b_series, b_shunt)
        plus optional trafo and shunt arrays.

    Returns
    -------
    y_bus : jnp.ndarray
        Complex128 array of shape (n_bus, n_bus).
    """
    n_bus = topology.n_bus
    y_bus = jnp.zeros((n_bus, n_bus), dtype=jnp.complex128)

    # ------------------------------------------------------------------
    # Lines (Pi-model)
    # ------------------------------------------------------------------
    y_series = params.g_series_pu + 1j * params.b_series_pu  # (n_line,)
    y_shunt_half = 0.5j * params.b_shunt_pu                  # (n_line,)

    def _stamp_line(carry: jnp.ndarray, idx: int) -> tuple[jnp.ndarray, None]:
        i = topology.from_bus[idx]
        j = topology.to_bus[idx]
        y = y_series[idx]
        yh = y_shunt_half[idx]
        m = carry
        m = m.at[i, i].add(y + yh)
        m = m.at[j, j].add(y + yh)
        m = m.at[i, j].add(-y)
        m = m.at[j, i].add(-y)
        return m, None

    # Python loop is fine here: topology is static, n_line is small for V1.
    for line_idx in range(topology.from_bus.shape[0]):
        y_bus = _stamp_line(y_bus, line_idx)[0]

    # ------------------------------------------------------------------
    # Transformers (off-nominal tap + phase shift Pi-model)
    # ------------------------------------------------------------------
    n_trafo = len(params.trafo_hv_bus)
    if n_trafo > 0 and params.trafo_tap_ratio is not None:
        for k in range(n_trafo):
            hv = params.trafo_hv_bus[k]
            lv = params.trafo_lv_bus[k]
            y_t = params.trafo_g_series_pu[k] + 1j * params.trafo_b_series_pu[k]
            y_m = params.trafo_g_mag_pu[k] + 1j * params.trafo_b_mag_pu[k]
            a = params.trafo_tap_ratio[k]           # tap magnitude
            phi = params.trafo_shift_rad[k]          # phase shift rad
            # complex tap: t = a * exp(j*phi)
            t = a * jnp.exp(1j * phi)
            t_conj = jnp.conj(t)
            a2 = (a * a).real
            y_bus = y_bus.at[hv, hv].add((y_t + y_m) / a2)
            y_bus = y_bus.at[lv, lv].add(y_t + y_m)
            y_bus = y_bus.at[hv, lv].add(-y_t / t_conj)
            y_bus = y_bus.at[lv, hv].add(-y_t / t)

    # ------------------------------------------------------------------
    # Shunts (diagonal only)
    # ------------------------------------------------------------------
    n_shunt = len(params.shunt_bus)
    if n_shunt > 0 and params.shunt_g_pu is not None:
        for k in range(n_shunt):
            bus_k = params.shunt_bus[k]
            y_sh = params.shunt_g_pu[k] + 1j * params.shunt_b_pu[k]
            y_bus = y_bus.at[bus_k, bus_k].add(y_sh)

    return y_bus
