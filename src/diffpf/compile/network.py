"""
Compile human-friendly NetworkSpec into static topology + differentiable params.

This module is the internal compiler used by ``diffpf.io.parser``.
Direct use is also fine for programmatic network construction (no JSON needed).
"""

from __future__ import annotations

import jax.numpy as jnp

from diffpf.core.types import CompiledTopology, NetworkParams, NetworkSpec


def _line_series_admittance(r_pu: float, x_pu: float) -> complex:
    z = complex(r_pu, x_pu)
    if abs(z) < 1e-12:
        raise ValueError("Line impedance must be non-zero (|z| < 1e-12 p.u.).")
    return 1.0 / z


def compile_network(spec: NetworkSpec) -> tuple[CompiledTopology, NetworkParams]:
    """
    Compile a ``NetworkSpec`` into JAX array structures.

    Parameters
    ----------
    spec : NetworkSpec
        Human-readable network definition with topology and per-unit parameters.
        May include optional ``trafos`` and ``shunts``.

    Returns
    -------
    topology : CompiledTopology
        Static index arrays (not differentiated).
    params : NetworkParams
        Differentiable float64 parameter arrays.
    """
    n_bus = len(spec.buses)
    if n_bus == 0:
        raise ValueError("NetworkSpec must contain at least one bus.")

    slack_indices = [i for i, bus in enumerate(spec.buses) if bus.is_slack]
    if len(slack_indices) != 1:
        raise ValueError(f"Exactly one slack bus required, found {len(slack_indices)}.")

    if len(spec.p_spec_pu) != n_bus or len(spec.q_spec_pu) != n_bus:
        raise ValueError("p_spec_pu and q_spec_pu length must equal number of buses.")

    slack_bus = slack_indices[0]
    variable_bus_indices = [i for i in range(n_bus) if i != slack_bus]
    variable_buses = jnp.asarray(variable_bus_indices, dtype=jnp.int32)

    # PV-Bus-Masken und Sollspannungen aus BusSpec ableiten
    is_pv_list = [spec.buses[i].is_pv for i in variable_bus_indices]
    v_set_list = [spec.buses[i].v_set_pu for i in variable_bus_indices]
    is_pv_mask = jnp.asarray(is_pv_list, dtype=bool)
    is_pq_mask = ~is_pv_mask
    v_set_pu_arr = jnp.asarray(v_set_list, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # Lines
    # ------------------------------------------------------------------
    from_bus, to_bus, g_series, b_series, b_shunt = [], [], [], [], []
    for line in spec.lines:
        if not (0 <= line.from_bus < n_bus and 0 <= line.to_bus < n_bus):
            raise ValueError(
                f"Line ({line.from_bus}→{line.to_bus}): endpoint out of range."
            )
        if line.from_bus == line.to_bus:
            raise ValueError("Self-loops are not supported.")

        y = _line_series_admittance(line.r_pu, line.x_pu)
        from_bus.append(line.from_bus)
        to_bus.append(line.to_bus)
        g_series.append(y.real)
        b_series.append(y.imag)
        b_shunt.append(line.b_shunt_pu)

    # ------------------------------------------------------------------
    # Transformers
    # ------------------------------------------------------------------
    t_hv: list[int] = []
    t_lv: list[int] = []
    t_g_series: list[float] = []
    t_b_series: list[float] = []
    t_g_mag: list[float] = []
    t_b_mag: list[float] = []
    t_tap: list[float] = []
    t_shift: list[float] = []

    for trafo in spec.trafos:
        if not (0 <= trafo.hv_bus < n_bus and 0 <= trafo.lv_bus < n_bus):
            raise ValueError(
                f"Trafo ({trafo.hv_bus}→{trafo.lv_bus}): endpoint out of range."
            )
        if trafo.hv_bus == trafo.lv_bus:
            raise ValueError("Transformer self-loop is not allowed.")
        import cmath
        z = complex(trafo.r_pu, trafo.x_pu)
        if abs(z) < 1e-12:
            raise ValueError(
                f"Trafo ({trafo.hv_bus}→{trafo.lv_bus}): near-zero series impedance."
            )
        y_t = 1.0 / z
        t_hv.append(trafo.hv_bus)
        t_lv.append(trafo.lv_bus)
        t_g_series.append(y_t.real)
        t_b_series.append(y_t.imag)
        t_g_mag.append(trafo.g_mag_pu)
        t_b_mag.append(trafo.b_mag_pu)
        t_tap.append(trafo.tap_ratio)
        t_shift.append(trafo.shift_rad)

    # ------------------------------------------------------------------
    # Shunts
    # ------------------------------------------------------------------
    sh_bus: list[int] = []
    sh_g: list[float] = []
    sh_b: list[float] = []

    for shunt in spec.shunts:
        if not (0 <= shunt.bus < n_bus):
            raise ValueError(f"Shunt bus {shunt.bus} out of range [0, {n_bus}).")
        sh_bus.append(shunt.bus)
        sh_g.append(shunt.g_pu)
        sh_b.append(shunt.b_pu)

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    topology = CompiledTopology(
        n_bus=n_bus,
        slack_bus=slack_bus,
        from_bus=jnp.asarray(from_bus, dtype=jnp.int32),
        to_bus=jnp.asarray(to_bus, dtype=jnp.int32),
        variable_buses=variable_buses,
        is_pq_mask=is_pq_mask,
        is_pv_mask=is_pv_mask,
    )
    params = NetworkParams(
        p_spec_pu=jnp.asarray(spec.p_spec_pu, dtype=jnp.float64),
        q_spec_pu=jnp.asarray(spec.q_spec_pu, dtype=jnp.float64),
        v_set_pu=v_set_pu_arr,
        g_series_pu=jnp.asarray(g_series, dtype=jnp.float64),
        b_series_pu=jnp.asarray(b_series, dtype=jnp.float64),
        b_shunt_pu=jnp.asarray(b_shunt, dtype=jnp.float64),
        slack_vr_pu=jnp.asarray(spec.slack_vr_pu, dtype=jnp.float64),
        slack_vi_pu=jnp.asarray(spec.slack_vi_pu, dtype=jnp.float64),
        # transformers
        trafo_g_series_pu=jnp.asarray(t_g_series, dtype=jnp.float64),
        trafo_b_series_pu=jnp.asarray(t_b_series, dtype=jnp.float64),
        trafo_g_mag_pu=jnp.asarray(t_g_mag, dtype=jnp.float64),
        trafo_b_mag_pu=jnp.asarray(t_b_mag, dtype=jnp.float64),
        trafo_tap_ratio=jnp.asarray(t_tap, dtype=jnp.float64),
        trafo_shift_rad=jnp.asarray(t_shift, dtype=jnp.float64),
        trafo_hv_bus=tuple(t_hv),
        trafo_lv_bus=tuple(t_lv),
        # shunts
        shunt_g_pu=jnp.asarray(sh_g, dtype=jnp.float64),
        shunt_b_pu=jnp.asarray(sh_b, dtype=jnp.float64),
        shunt_bus=tuple(sh_bus),
    )
    return topology, params
