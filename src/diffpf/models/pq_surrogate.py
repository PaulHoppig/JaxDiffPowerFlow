"""Small JAX-only neural PQ surrogate for upstream weather coupling.

The model is intentionally compact and dependency-free. It maps weather inputs
``[g_poa_wm2, t_amb_c, wind_ms]`` to an active-power injection and derives
reactive power from the fixed ratio ``Q = kappa * P``. It is a modular
replacement for the analytical PV weather model, not a voltage-regulating PV
bus model.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from diffpf.models.pv import PV_BASE_P_MW, PV_Q_OVER_P, PVInjection


class MLPParams(NamedTuple):
    """Weights and biases of a fully connected MLP."""

    weights: tuple[jnp.ndarray, ...]
    biases: tuple[jnp.ndarray, ...]


class WeatherInputNormalization(NamedTuple):
    """Affine weather-input normalization in physical units."""

    center: jnp.ndarray = jnp.asarray((600.0, 17.5, 5.25), dtype=jnp.float64)
    scale: jnp.ndarray = jnp.asarray((600.0, 27.5, 4.75), dtype=jnp.float64)


class SurrogateTrainingConfig(NamedTuple):
    """Small default training configuration for the Exp. 4 distillation run."""

    seed: int = 42
    train_samples: int = 512
    val_samples: int = 128
    hidden_width: int = 8
    hidden_layers: int = 2
    learning_rate: float = 0.03
    max_train_steps: int = 1200
    log_every: int = 50


DEFAULT_WEATHER_NORMALIZATION = WeatherInputNormalization()
DEFAULT_TRAINING_CONFIG = SurrogateTrainingConfig()


def init_mlp_params(
    key: jax.Array,
    input_dim: int = 3,
    hidden_width: int = 8,
    hidden_layers: int = 2,
    output_dim: int = 1,
) -> MLPParams:
    """Initialize a small tanh MLP with deterministic JAX PRNG keys."""

    layer_dims = [input_dim] + [hidden_width] * hidden_layers + [output_dim]
    keys = jax.random.split(key, len(layer_dims) - 1)
    weights = []
    biases = []
    for subkey, in_dim, out_dim in zip(keys, layer_dims[:-1], layer_dims[1:]):
        limit = jnp.sqrt(6.0 / (in_dim + out_dim))
        weights.append(
            jax.random.uniform(
                subkey,
                shape=(in_dim, out_dim),
                minval=-limit,
                maxval=limit,
                dtype=jnp.float64,
            )
        )
        biases.append(jnp.zeros((out_dim,), dtype=jnp.float64))
    return MLPParams(weights=tuple(weights), biases=tuple(biases))


def normalize_weather_inputs(
    x: jnp.ndarray,
    norm: WeatherInputNormalization = DEFAULT_WEATHER_NORMALIZATION,
) -> jnp.ndarray:
    """Normalize weather inputs ``[..., g_poa_wm2, t_amb_c, wind_ms]``."""

    values = jnp.asarray(x, dtype=jnp.float64)
    return (values - norm.center) / norm.scale


def mlp_apply(params: MLPParams, x_norm: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the MLP on normalized inputs.

    The returned array has the same leading dimensions as ``x_norm`` and a
    final dimension equal to the output dimension. Hidden layers use ``tanh``.
    """

    y = jnp.asarray(x_norm, dtype=jnp.float64)
    for weight, bias in zip(params.weights[:-1], params.biases[:-1]):
        y = jnp.tanh(y @ weight + bias)
    return y @ params.weights[-1] + params.biases[-1]


def neural_pq_injection_from_weather(
    params: MLPParams,
    norm: WeatherInputNormalization,
    g_poa_wm2: jnp.ndarray,
    t_amb_c: jnp.ndarray,
    wind_ms: jnp.ndarray,
    *,
    p_ref_mw: float = PV_BASE_P_MW,
    kappa: float = PV_Q_OVER_P,
) -> PVInjection:
    """Return the P-only neural PQ injection from weather inputs.

    The network predicts normalized active power ``P_nn / p_ref_mw``. Reactive
    power is deterministically derived as ``Q_nn = kappa * P_nn`` to match the
    replaced static generator's baseline ratio.
    """

    x = jnp.stack(
        [
            jnp.asarray(g_poa_wm2, dtype=jnp.float64),
            jnp.asarray(t_amb_c, dtype=jnp.float64),
            jnp.asarray(wind_ms, dtype=jnp.float64),
        ],
        axis=-1,
    )
    p_norm = jnp.squeeze(mlp_apply(params, normalize_weather_inputs(x, norm)), axis=-1)
    p_nn_mw = jnp.asarray(p_ref_mw, dtype=jnp.float64) * p_norm
    q_nn_mvar = jnp.asarray(kappa, dtype=jnp.float64) * p_nn_mw
    return PVInjection(p_pv_mw=p_nn_mw, q_pv_mvar=q_nn_mvar)


def count_mlp_parameters(params: MLPParams) -> int:
    """Return the total number of scalar trainable parameters."""

    return int(
        sum(weight.size for weight in params.weights)
        + sum(bias.size for bias in params.biases)
    )
