"""
Y-bus matrix assembly via the stamping method.

Each Pi-model line contributes four stamps to Y_bus:
  Y[i,i] += y_series + y_shunt/2
  Y[j,j] += y_series + y_shunt/2
  Y[i,j] -= y_series
  Y[j,i] -= y_series

where y_series = 1 / (r + jx)  and  y_shunt = j * b_shunt.

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
        Differentiable line parameters (g_series, b_series, b_shunt).

    Returns
    -------
    y_bus : jnp.ndarray
        Complex128 array of shape (n_bus, n_bus).
    """
    n_bus = topology.n_bus
    y_bus = jnp.zeros((n_bus, n_bus), dtype=jnp.complex128)

    y_series = params.g_series_pu + 1j * params.b_series_pu  # shape (n_line,)
    y_shunt_half = 0.5j * params.b_shunt_pu                  # shape (n_line,)

    def _stamp(carry: jnp.ndarray, idx: int) -> tuple[jnp.ndarray, None]:
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
    # For large networks, replace with jax.lax.scan over a fixed-size range.
    for line_idx in range(topology.from_bus.shape[0]):
        y_bus = _stamp(y_bus, line_idx)[0]

    return y_bus
