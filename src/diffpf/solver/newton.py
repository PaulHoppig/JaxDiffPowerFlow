"""Newton solver for the rectangular AC power-flow residual."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from diffpf.core.residuals import power_flow_residual, residual_loss
from diffpf.core.types import CompiledTopology, NetworkParams, PFState


@dataclass(frozen=True)
class NewtonOptions:
    """Configuration for the damped Newton solver."""

    max_iters: int = 25
    tolerance: float = 1e-10
    damping: float = 1.0


@dataclass(frozen=True)
class NewtonResult:
    """Structured Newton solver output including convergence metadata."""

    solution: PFState
    residual_norm: jnp.ndarray
    loss: jnp.ndarray
    iterations: jnp.ndarray
    converged: jnp.ndarray


def _residual_from_vector(
    vector: jnp.ndarray,
    topology: CompiledTopology,
    params: NetworkParams,
) -> jnp.ndarray:
    state = PFState.from_vector(vector, topology.variable_buses.shape[0])
    return power_flow_residual(topology, params, state)


def solve_power_flow(
    topology: CompiledTopology,
    params: NetworkParams,
    initial_state: PFState,
    options: NewtonOptions = NewtonOptions(),
) -> tuple[PFState, jnp.ndarray, jnp.ndarray]:
    """Solve the PQ residual with a damped Newton method."""
    result = solve_power_flow_result(topology, params, initial_state, options)
    return result.solution, result.residual_norm, result.loss


def solve_power_flow_result(
    topology: CompiledTopology,
    params: NetworkParams,
    initial_state: PFState,
    options: NewtonOptions = NewtonOptions(),
) -> NewtonResult:
    """Solve the PQ residual and return convergence metadata."""

    initial_vector = initial_state.as_vector()
    n_var = topology.variable_buses.shape[0]

    def cond_fn(carry: tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> bool:
        iteration, _, residual_norm, _ = carry
        return jnp.logical_and(iteration < options.max_iters, residual_norm > options.tolerance)

    def body_fn(
        carry: tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        iteration, vector, _, _ = carry
        residual = _residual_from_vector(vector, topology, params)
        jacobian = jax.jacfwd(_residual_from_vector)(vector, topology, params)
        step = jnp.linalg.solve(jacobian, residual)
        next_vector = vector - options.damping * step
        next_residual = _residual_from_vector(next_vector, topology, params)
        next_norm = jnp.linalg.norm(next_residual, ord=2)
        return iteration + 1, next_vector, next_norm, residual_loss(
            topology,
            params,
            PFState.from_vector(next_vector, n_var),
        )

    initial_residual = _residual_from_vector(initial_vector, topology, params)
    initial_norm = jnp.linalg.norm(initial_residual, ord=2)
    initial_loss = residual_loss(topology, params, initial_state)
    iterations, solution_vector, residual_norm, loss = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (0, initial_vector, initial_norm, initial_loss),
    )
    return NewtonResult(
        solution=PFState.from_vector(solution_vector, n_var),
        residual_norm=residual_norm,
        loss=loss,
        iterations=iterations,
        converged=residual_norm <= options.tolerance,
    )
