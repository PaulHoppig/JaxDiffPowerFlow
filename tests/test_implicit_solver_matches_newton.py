"""Forward consistency tests for the implicit power-flow solver."""

from __future__ import annotations

import jax
import numpy as np

from diffpf.core import power_flow_residual, state_to_voltage
from diffpf.solver import NewtonOptions, solve_power_flow, solve_power_flow_implicit


def test_implicit_solver_matches_newton_state(three_bus_case):
    topology, params, state = three_bus_case
    options = NewtonOptions(max_iters=30, tolerance=1e-10, damping=1.0)

    newton_solution, newton_norm, _ = solve_power_flow(topology, params, state, options)
    implicit_solution = solve_power_flow_implicit(topology, params, state, options)

    np.testing.assert_allclose(
        np.asarray(implicit_solution.as_vector()),
        np.asarray(newton_solution.as_vector()),
        rtol=1e-10,
        atol=1e-10,
    )
    implicit_norm = np.linalg.norm(np.asarray(power_flow_residual(topology, params, implicit_solution)))
    assert implicit_norm < 1e-8
    assert float(newton_norm) < 1e-8


def test_implicit_solver_is_jittable(three_bus_case):
    topology, params, state = three_bus_case
    solution = jax.jit(solve_power_flow_implicit)(topology, params, state)
    voltage = state_to_voltage(topology, params, solution)

    assert voltage.shape == (topology.n_bus,)
    assert np.all(np.isfinite(np.asarray(voltage)))
