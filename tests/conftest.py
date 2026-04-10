"""
Pytest-Konfiguration und gemeinsame Fixtures für die diffpf-Testsuite.

Alle Import-Pfade werden über pyproject.toml [tool.pytest.ini_options]
pythonpath aufgelöst — keine manuelle sys.path-Manipulation nötig.

Fixtures
--------
three_bus_case
    Lädt das 3-Bus-Demonstrationsnetz aus der JSON-Datei.
    Scope: session (einmal pro Test-Lauf, nicht per Test).

solved_three_bus
    Löst das 3-Bus-Netz mit Newton-Raphson.
    Scope: session.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from diffpf.io import load_network
from diffpf.solver import NewtonOptions, solve_power_flow

CASES_DIR = Path(__file__).resolve().parents[1] / "cases"


@pytest.fixture(scope="session")
def three_bus_case():
    """Liefert (topology, params, state) für das 3-Bus-Netz aus JSON."""
    return load_network(CASES_DIR / "three_bus_poc.json")


@pytest.fixture(scope="session")
def solved_three_bus(three_bus_case):
    """Liefert (topology, params, solution, residual_norm) nach Newton-Raphson."""
    topology, params, state = three_bus_case
    solution, norm, loss = solve_power_flow(
        topology,
        params,
        state,
        NewtonOptions(max_iters=30, tolerance=1e-10, damping=1.0),
    )
    return topology, params, solution, norm, loss
