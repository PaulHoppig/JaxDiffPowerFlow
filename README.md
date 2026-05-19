# diffpf

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-differentiable%20power%20flow-orange.svg)](https://jax.readthedocs.io/)
[![Status](https://img.shields.io/badge/status-research%20prototype-lightgrey.svg)](#project-status)

**Differentiable AC power flow in JAX for coupling electrical grid physics with
upstream models.**

`diffpf` is a research prototype for stationary AC power-flow calculations that
can be embedded into larger differentiable model chains. Its focus is not to
replace established tools such as `pandapower`, but to show that grid physics
can be implemented as a clean, validated, differentiable layer.

The central use case is an end-to-end chain such as:

```text
weather -> PV model -> P/Q injection -> AC power flow -> voltages, losses, export
```

This makes it possible to compute sensitivities of electrical quantities with
respect to non-electrical upstream inputs, for example irradiance, ambient
temperature, wind speed, PV model parameters, or curtailment factors.

## Highlights

- Rectangular, complex AC power-flow core in **JAX** with 64-bit precision.
- Newton-Raphson forward solve for stationary AC operating points.
- Implicit differentiation through the converged root problem via
  `jax.lax.custom_root`, instead of differentiating through every Newton step.
- Clean separation between external I/O, static compilation, numerical core,
  validation, upstream models, and experiments.
- Controlled `pandapower` import and validation pipeline for the supported
  element subset.
- Reproducible experiment artifacts as CSV/JSON, plus publication-oriented
  figures for the bachelor thesis workflow.

## What This Project Is

`diffpf` is:

- a differentiable stationary AC power-flow demonstrator,
- a physics layer for cross-domain sensitivity analysis,
- a compact test bed for coupling analytical and neural upstream models,
- a basis for simple gradient-based inverse or optimization tasks.

`diffpf` is not:

- a full industrial grid analysis package,
- a complete `pandapower` reimplementation,
- a general OPF solver,
- a claim of superiority over classical PF/OPF tools,
- a real-time or large-grid benchmark.

## Current Scope

The implemented model scope covers a stationary, balanced AC formulation with:

- one slack bus,
- PQ buses,
- idealized PV-bus residual support in the core,
- fixed P/Q loads and injections,
- line Pi models,
- 2-winding transformers with tap and phase shift,
- shunts,
- static topology,
- per-unit internal calculations,
- differentiable continuous parameters in `NetworkParams`.

The controlled `pandapower` adapter supports the subset needed by the current
experiments, including `bus`, `ext_grid`, `load`, `sgen`, simplified `gen`,
`line`, 2-winding `trafo`, `shunt`, and selected switch preprocessing such as
closed bus-bus switch fusion.

Known non-goals and limitations include controller logic, Q limits, PV-to-PQ
switching, 3-winding transformers, unsymmetric or three-phase power flow,
complete thermal equipment assessment, and full `pandapower` feature parity.

## Scientific Demonstrator

The original 3-bus case in `cases/three_bus_poc.json` remains as a minimal
regression and sanity-check network.

The main demonstrator for the later experiments is:

```python
pandapower.networks.example_simple()
```

For strict numerical comparisons, `diffpf` uses a **scope-matched** version of
this network: the active `gen` is treated as a fixed P injection with `Q = 0`,
consistent with the current differentiable model scope. The original
`pandapower` PV-bus behavior is kept as context, not as a strict equality
target.

The upstream PV coupling point is:

```text
bus:              "MV Bus 2"
replaced element: sgen "static generator"
reference P/Q:    2.0 MW / -0.5 MVAr
Q/P ratio:       -0.25
```

The PV plant is modeled as a weather-dependent PQ injection, not as a
voltage-regulating PV bus.

## Method

The electrical equations use the standard complex AC form:

```text
I = Y_bus @ V
S = V * conj(I)
```

Internally, voltages are complex, while the free solver state is real:

```text
x = [V_r | V_i]
```

For PQ buses, the residuals are:

```text
r_P = P_spec - Re(S_calc)
r_Q = Q_spec - Im(S_calc)
```

The operating point is solved with Newton-Raphson. Gradients through the solved
power flow are computed by implicit differentiation of the converged root
problem. This keeps the differentiated object close to the physical stationary
model rather than to the details of a particular iterative trajectory.

## Repository Layout

```text
src/diffpf/
  core/        numerical types, per-unit helpers, Y-bus, residuals, observables
  compile/     NetworkSpec -> CompiledTopology + NetworkParams
  solver/      Newton solver and implicit differentiation
  io/          JSON and pandapower adapters
  models/      JAX-compatible upstream PV and surrogate models
  validation/  finite differences and pandapower reference checks
  numerics/    compatibility re-exports
  pipeline/    reserved for end-to-end pipelines
  viz/         reserved for visualization helpers

cases/         small canonical network definitions
experiments/   reproducible scripts for the thesis experiments
tests/         pytest suite for core behavior and experiment artifacts
docs/          architecture notes, context documents, plans, and status notes
```

The most important design rule is that `core/`, `compile/`, and `solver/` stay
free of `pandapower`, JSON parsing, and file I/O logic.

## Implemented Experiments

The `CHANGELOG.md` is the authoritative source for what has actually been
implemented. At the current repository state, the implemented experiment chain
is:

| Experiment | Purpose | Main artifacts |
| --- | --- | --- |
| Exp. 1 | Forward-solve validation against `pandapower`, including the 3-bus PoC and `example_simple()` | `experiments/results/exp01_example_simple_validation/` |
| Exp. 2 | Implicit AD gradients vs. central finite differences | `experiments/results/exp02_example_simple_gradients/` |
| Exp. 3 | Cross-domain weather -> PV -> grid sensitivities | `experiments/results/exp03_cross_domain_pv_weather/` |
| Exp. 4 | Modular upstream coupling with analytical PV, a JAX-only NN surrogate, and a direct baseline | `experiments/results/exp04_modular_upstream_nn_surrogate/` |
| Exp. 5a | Network screening for PV-curtailment preparation | `experiments/results/exp05a_network_screening/` |
| Exp. 5b | Gradient-based PV-curtailment optimization for one selected operating point | `experiments/results/exp05b_optimize_pv_curtailment/` |

Several plot-only pipelines generate PNG/PDF figures from existing CSV/JSON
artifacts without rerunning the numerical experiments.

## Key Results

Highlights from the current artifacts and changelog:

- The original 3-bus PoC matches `pandapower` within numerical round-off.
- `example_simple()` is imported, compiled, solved, and validated in a
  scope-matched mode across seven scenarios.
- Experiment 2 validates 48 implicit gradients against central finite
  differences on `example_simple()`.
- Experiment 3 runs 108 weather/PV/grid forward cases and exports 1296
  end-to-end weather sensitivities.
- Experiment 4 trains a small JAX-only MLP surrogate on synthetic PV-weather
  data and couples it through the same P/Q interface as the analytical PV model.
- Experiment 5b solves a one-dimensional PV-curtailment demonstration:
  the selected full-PV case exports `7.599971 MW`; the optimized curtailment
  factor is `0.714203`, reaching `6.990006 MW` export against an internal
  `7.0 MW` target.

These results are local, demonstrator-based validation results. They are not a
general proof for arbitrary networks, operating points, controllers, or
discrete switching logic.

## Installation

Create and activate a virtual environment, then install the project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For the full experiment and plotting environment, install the manual dependency
list as well:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

The project is configured for Python 3.10 or newer.

## Quickstart

Solve the minimal 3-bus JSON case:

```python
from diffpf.io import load_network
from diffpf.solver import solve_power_flow_result
from diffpf.core.observables import power_flow_observables

topology, params, initial_state = load_network("cases/three_bus_poc.json")

result = solve_power_flow_result(topology, params, initial_state)
obs = power_flow_observables(topology, params, result.solution)

print("converged:", bool(result.converged))
print("residual norm:", float(result.residual_norm))
print("non-slack voltage magnitudes:", obs.voltage_mag_pu)
print("slack P/Q [pu]:", obs.slack_p_pu, obs.slack_q_pu)
```

For the current public API, prefer importing from `diffpf.core`, `diffpf.io`,
`diffpf.solver`, `diffpf.models`, and `diffpf.validation`. The `numerics/`
module is kept mainly for backward-compatible re-exports.

## Reproducing Experiments

Run the scripts directly from the repository root:

```powershell
python experiments/exp01_validate_example_simple.py
python experiments/exp02_validate_gradients_example_simple.py
python experiments/exp03_cross_domain_pv_weather.py
python experiments/exp04_modular_upstream_nn_surrogate.py
python experiments/exp05a_network_screening.py
python experiments/exp05b_optimize_pv_curtailment.py
```

Generate figures from existing artifacts:

```powershell
python experiments/plot_exp01_validation_figures.py
python experiments/plot_exp02_gradient_figures.py
python experiments/plot_exp03_figures.py
python experiments/plot_exp04_training_figures.py
python experiments/plot_exp05_figures.py
```

Some experiments are heavier than unit tests because they solve many nonlinear
power flows, train a small surrogate, or regenerate report artifacts.

## Testing

Run the test suite with:

```powershell
python -m pytest
```

Useful focused checks:

```powershell
python -m pytest tests/test_newton.py tests/test_residuals.py tests/test_ybus.py
python -m pytest tests/test_implicit_gradients_vs_fd.py
python -m pytest tests/test_pv_model.py tests/test_pq_surrogate_model.py
python -m pytest tests/test_exp05b_optimize_pv_curtailment_outputs.py
```

The pytest configuration in `pyproject.toml` adds `src/` and the repository root
to the import path.

## Development Guardrails

When extending the project:

- keep `pandapower` logic out of `core/`, `compile/`, and `solver/`,
- use `jax.numpy` in numerical hot paths,
- avoid Python control flow on JAX tracer values,
- keep topology changes outside the differentiated solve,
- put differentiable continuous quantities into `NetworkParams`,
- add tests and a changelog entry for new model behavior,
- document known simplifications in experiment metadata and result READMEs.

## Project Status

`diffpf` is a bachelor-thesis research prototype. The core solver,
`pandapower`-oriented I/O pipeline, validation experiments, PV upstream model,
NN surrogate coupling, and a small curtailment optimization demonstration are
implemented.

The most important remaining limitations are full `pandapower` generator
semantics, Q limits, PV-to-PQ switching, controller logic, detailed transformer
parity, and validation beyond the current demonstrator networks.
