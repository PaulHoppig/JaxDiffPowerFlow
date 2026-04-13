"""Implicitly differentiated power-flow solve via ``jax.lax.custom_root``."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from diffpf.core.residuals import power_flow_residual, residual_loss
from diffpf.core.types import CompiledTopology, NetworkParams, PFState
from diffpf.solver.newton import NewtonOptions, solve_power_flow


@dataclass(frozen=True)
class ImplicitPowerFlowResult:
    """Implicit solver output with the same physical solution as Newton."""

    solution: PFState
    residual_norm: jnp.ndarray
    loss: jnp.ndarray


def _state_from_vector(vector: jnp.ndarray, topology: CompiledTopology) -> PFState:
    return PFState.from_vector(vector, topology.variable_buses.shape[0])


def solve_power_flow_implicit(
    topology: CompiledTopology,
    params: NetworkParams,
    initial_state: PFState,
    options: NewtonOptions = NewtonOptions(),
) -> PFState:
    """Solve the AC power-flow equations with implicit differentiation.

    The forward solve still uses the existing Newton implementation. The
    derivative, however, is supplied by ``custom_root`` and therefore uses the
    linearized residual at the converged operating point instead of unrolling
    Newton iterations.
    """

    n_state = initial_state.as_vector().shape[0]

    def residual_from_vector(vector: jnp.ndarray) -> jnp.ndarray:
        return power_flow_residual(topology, params, _state_from_vector(vector, topology))

    def solve_fn(_residual_fn, initial_vector: jnp.ndarray) -> jnp.ndarray:
        initial = _state_from_vector(initial_vector, topology)
        solution, _, _ = solve_power_flow(topology, params, initial, options)
        return solution.as_vector()

    def tangent_solve_fn(linearized_residual, cotangent: jnp.ndarray) -> jnp.ndarray:
        # Build the small dense Jacobian from the linear map supplied by JAX.
        basis = jnp.eye(n_state, dtype=cotangent.dtype)
        jacobian = jax.vmap(linearized_residual)(basis).T
        return jnp.linalg.solve(jacobian, cotangent)

    solution_vector = jax.lax.custom_root(
        residual_from_vector,
        initial_state.as_vector(),
        solve_fn,
        tangent_solve_fn,
    )
    return _state_from_vector(solution_vector, topology)


def solve_power_flow_implicit_result(
    topology: CompiledTopology,
    params: NetworkParams,
    initial_state: PFState,
    options: NewtonOptions = NewtonOptions(),
) -> ImplicitPowerFlowResult:
    """Solve with implicit differentiation and return residual diagnostics."""

    solution = solve_power_flow_implicit(topology, params, initial_state, options)
    residual = power_flow_residual(topology, params, solution)
    norm = jnp.linalg.norm(residual, ord=2)
    return ImplicitPowerFlowResult(
        solution=solution,
        residual_norm=norm,
        loss=residual_loss(topology, params, solution),
    )
