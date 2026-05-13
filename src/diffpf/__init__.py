"""
diffpf – Differentiable AC Power Flow in JAX.

Package structure
-----------------
core/       Pure numerical primitives (types, units, Y-bus, residuals).
io/         JSON loading and parsing (the only bridge to dict/file world).
compile/    Internal compiler: NetworkSpec → JAX arrays.
solver/     Newton-Raphson solver with jax.lax.while_loop.
models/     Backward-compat re-exports from core/ (deprecated, keep for now).
numerics/   Backward-compat re-exports from core/ (deprecated, keep for now).
validation/ Gradient-check utilities (finite differences).
pipeline/   End-to-end differentiable graphs (V2+).
viz/        Matplotlib helpers.
"""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
