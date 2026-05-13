"""Tests for the JAX-only neural PQ surrogate model."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np

from diffpf.models.pq_surrogate import (
    DEFAULT_WEATHER_NORMALIZATION,
    count_mlp_parameters,
    init_mlp_params,
    neural_pq_injection_from_weather,
)


def test_parameter_initialization_is_deterministic_for_fixed_key():
    p1 = init_mlp_params(jax.random.PRNGKey(0))
    p2 = init_mlp_params(jax.random.PRNGKey(0))

    for w1, w2 in zip(p1.weights, p2.weights):
        np.testing.assert_allclose(np.asarray(w1), np.asarray(w2))
    for b1, b2 in zip(p1.biases, p2.biases):
        np.testing.assert_allclose(np.asarray(b1), np.asarray(b2))


def test_model_output_is_scalar_and_finite():
    params = init_mlp_params(jax.random.PRNGKey(1))
    injection = neural_pq_injection_from_weather(
        params,
        DEFAULT_WEATHER_NORMALIZATION,
        800.0,
        25.0,
        2.0,
    )

    assert injection.p_pv_mw.shape == ()
    assert injection.q_pv_mvar.shape == ()
    assert jnp.isfinite(injection.p_pv_mw)
    assert jnp.isfinite(injection.q_pv_mvar)


def test_p_only_model_uses_fixed_q_over_p_ratio():
    params = init_mlp_params(jax.random.PRNGKey(2))
    injection = neural_pq_injection_from_weather(
        params,
        DEFAULT_WEATHER_NORMALIZATION,
        900.0,
        20.0,
        3.0,
        kappa=-0.25,
    )

    if abs(float(injection.p_pv_mw)) > 1e-12:
        np.testing.assert_allclose(
            float(injection.q_pv_mvar / injection.p_pv_mw),
            -0.25,
            rtol=1e-12,
            atol=1e-12,
        )


def test_jit_works_for_inference():
    params = init_mlp_params(jax.random.PRNGKey(3))

    @jax.jit
    def fn(g, t, w):
        return neural_pq_injection_from_weather(
            params,
            DEFAULT_WEATHER_NORMALIZATION,
            g,
            t,
            w,
        ).p_pv_mw

    value = fn(jnp.asarray(800.0), jnp.asarray(25.0), jnp.asarray(2.0))
    assert jnp.isfinite(value)


def test_gradients_with_respect_to_weather_are_finite():
    params = init_mlp_params(jax.random.PRNGKey(4))

    def fn(g, t, w):
        return neural_pq_injection_from_weather(
            params,
            DEFAULT_WEATHER_NORMALIZATION,
            g,
            t,
            w,
        ).p_pv_mw

    grads = jax.grad(fn, argnums=(0, 1, 2))(
        jnp.asarray(800.0),
        jnp.asarray(25.0),
        jnp.asarray(2.0),
    )
    assert all(jnp.isfinite(grad) for grad in grads)


def test_parameter_count_stays_small():
    params = init_mlp_params(jax.random.PRNGKey(5))

    assert count_mlp_parameters(params) < 500


def test_surrogate_module_does_not_import_pandapower():
    import diffpf.models.pq_surrogate as module

    source = inspect.getsource(module)
    assert "pandapower" not in source
