# Architektur und Dateiübersicht

Dieses Dokument beschreibt Zweck und Verantwortlichkeiten aller Dateien im Projekt.

---

## Wurzelverzeichnis

| Datei | Aufgabe |
|---|---|
| `pyproject.toml` | Build-Konfiguration (setuptools), Abhängigkeiten, pytest-Einstellungen, Ruff/Black-Konfiguration. Einzige Quelle der Wahrheit für Projektmetadaten. |
| `requirements.txt` | Manuelle Abhängigkeitsliste für direkte pip-Installationen (alternativ zu pyproject.toml). |
| `.gitignore` | Definiert welche Dateien Git ignoriert (`.venv/`, `__pycache__/`, `.egg-info/` usw.). |
| `README.md` | Projektübersicht, Installationsanleitung, Schnellstart-Beispiel. |
| `AGENTS.md` | Anweisungen für KI-Agenten, die an diesem Repo arbeiten. |

---

## `src/diffpf/` — Python-Paket

### `src/diffpf/__init__.py`
Paket-Einstiegspunkt. Aktiviert `jax_enable_x64`, damit alle Berechnungen in
float64 stattfinden (notwendig für numerische Genauigkeit im Leistungsfluss).

---

### `src/diffpf/core/` — Reiner numerischer Kern

Dieses Modul enthält ausschließlich reine Funktionen und Datenstrukturen.
**Invariante:** `core/` importiert niemals aus `io/`, `compile/`, oder
dict/JSON-Logik. Alle Funktionen sind JIT- und grad-fähig.

| Datei | Aufgabe |
|---|---|
| `__init__.py` | Re-exportiert alle öffentlichen Symbole aus `types`, `units`, `ybus`, `residuals`. Einziger nötiger Import für Nutzer des Kerns. |
| `types.py` | Definiert alle zentralen Datentypen als gefrorene Dataclasses, die als JAX-Pytrees registriert sind. `BusSpec`/`LineSpec`/`NetworkSpec` sind menschenlesbare Eingabestrukturen. `CompiledTopology` enthält statische Topologie-Arrays (Indizes, Busnummern), `NetworkParams` die differenzierbaren physikalischen Parameter, `PFState` den rechteckigen Spannungszustand der Nicht-Slack-Busse. Der Trick mit `meta_fields`/`data_fields` trennt explizit statische (nicht differenzierbare) von dynamischen (differenzierbaren) Blättern im Pytree. |
| `units.py` | Klasse `BaseValues`: kapselt die Systembasis (S_base in MVA, V_base in kV) und bietet bidirektionale Konversionsmethoden zwischen physikalischen Einheiten (MW, MVAR, kV, Ω) und Per-Unit-Größen. Enthält keine JAX-Abhängigkeit. |
| `ybus.py` | Funktion `build_ybus()`: baut die komplexe Busdmittanzmatrix Y_bus durch das Stamping-Verfahren auf. Jede Pi-Modell-Leitung stempelt vier Einträge: Diagonal- und Off-Diagonal-Elemente für Serien- und Shunt-Anteile. Gibt eine (n_bus × n_bus) complex128-Matrix zurück. |
| `residuals.py` | Kernfunktionen des Leistungsfluss-Residuums: `state_to_voltage()` rekonstruiert den vollständigen Spannungsvektor (inkl. Slack-Bus), `calc_power_injection()` berechnet S = V · conj(Y·V), `power_flow_residual()` liefert den concatenierten P/Q-Mismatch-Vektor für Nicht-Slack-Busse, `residual_loss()` berechnet den skalaren Verlust 0,5·‖r‖². |

---

### `src/diffpf/io/` — JSON-Laden und Parsen

Die einzige Brücke zwischen der menschenlesbaren JSON-Welt und JAX-Arrays.
`core/` kennt keine Dict-Logik.

| Datei | Aufgabe |
|---|---|
| `__init__.py` | Re-exportiert `load_json`, `load_network`, `parse`, `RawNetwork`. |
| `reader.py` | **Schicht 1 (rein Python, kein JAX):** Lädt eine JSON-Netz-Datei und überführt sie in typisierte Dataclasses (`RawBase`, `RawBus`, `RawLine`, `RawNetwork`). Validiert semantische Invarianten: genau ein Slack-Bus, keine doppelten IDs, keine Null-Impedanzen, gültige Leitungsendpunkte. Wirft `ValueError` bei jeder Verletzung — bevor JAX-Code berührt wird. |
| `parser.py` | **Schicht 2 (Brücke zu JAX):** Nimmt ein `RawNetwork` entgegen, konvertiert physikalische Einheiten via `BaseValues` in Per-Unit, baut eine kontinuierliche Bus-Index-Map (externe IDs → 0-basierte Indizes), zerlegt die Slack-Spannung in Rechteckkoordinaten und ruft `compile_network()` auf. Gibt `(CompiledTopology, NetworkParams, PFState)` zurück. `PFState` ist dabei ein Flat-Start (Vr = 1, Vi = 0). Öffentliche Schnittstelle: `load_network(path)` als One-Liner. |

---

### `src/diffpf/compile/` — Interner Netz-Compiler

| Datei | Aufgabe |
|---|---|
| `__init__.py` | Re-exportiert `compile_network`. |
| `network.py` | Funktion `compile_network()`: überführt ein `NetworkSpec` (Python-Datenstruktur) in das JAX-Array-Paar `(CompiledTopology, NetworkParams)`. Berechnet Serienadmittanzen (y = 1/z), prüft Gültigkeitsbedingungen und baut alle JAX-Arrays in float64 auf. Wird intern von `io/parser.py` aufgerufen, kann aber auch direkt für programmatische Netzdefinitionen verwendet werden. |

---

### `src/diffpf/solver/` — Newton-Raphson-Solver

| Datei | Aufgabe |
|---|---|
| `__init__.py` | Re-exportiert `NewtonOptions`, `solve_power_flow`. |
| `newton.py` | Implementiert den gedämpften Newton-Raphson-Solver über `jax.lax.while_loop` (JIT-kompatibel). `NewtonOptions` konfiguriert Iterationslimit, Konvergenztoleranz und Dämpfungsfaktor. In jedem Schritt wird der Jacobi via `jax.jacfwd` berechnet und das Gleichungssystem mit `jnp.linalg.solve` gelöst. Gibt `(PFState, residual_norm, loss)` zurück. |

---

### `src/diffpf/models/` — Backward-Compatibility-Shim *(veraltet)*

| Datei | Aufgabe |
|---|---|
| `__init__.py` | Re-exportiert alle Typen aus `diffpf.core.types`. |
| `network.py` | Weiterleitungs-Shim: importiert und re-exportiert aus `core.types`. Bleibt solange erhalten, bis alle externen Imports auf `diffpf.core` umgestellt sind. |

---

### `src/diffpf/numerics/` — Backward-Compatibility-Shim *(veraltet)*

| Datei | Aufgabe |
|---|---|
| `__init__.py` | Re-exportiert alle Funktionen aus `diffpf.core.ybus` und `diffpf.core.residuals`. |
| `power_flow.py` | Weiterleitungs-Shim: importiert und re-exportiert aus `core.ybus` und `core.residuals`. |

---

### `src/diffpf/validation/` — Gradienten-Validierung

| Datei | Aufgabe |
|---|---|
| `__init__.py` | Re-exportiert `central_difference`. |
| `finite_diff.py` | Hilfsfunktion `central_difference()`: berechnet den skalaren zentralen Finite-Differenzen-Gradienten einer Funktion f: ℝ → ℝ. Wird in Tests verwendet, um Autodiff-Gradienten gegen numerische Näherungen zu prüfen. |

---

### `src/diffpf/pipeline/` — End-to-End-Rechengraph *(V2+)*

| Datei | Aufgabe |
|---|---|
| `__init__.py` | Platzhalter. Hier entstehen in V2 differenzierbare Pipelines wie Wetter → PV-Modell → Leistungsfluss → Netzgrößen. |

---

### `src/diffpf/viz/` — Visualisierung *(V2+)*

| Datei | Aufgabe |
|---|---|
| `__init__.py` | Platzhalter. Hier entstehen Matplotlib-Hilfsfunktionen für Spannungsprofile und Konvergenzplots. |

---

## `cases/` — Netzdefinitionen

| Datei | Aufgabe |
|---|---|
| `three_bus_poc.json` | **Kanonische Netzdefinition** des 3-Bus-Demonstrators als menschenlesbare JSON-Datei. Enthält Systembasis, drei Busse (Umspannwerk/Slack, Wohngebiet/Last, PV-Park/Einspeisung) und drei Pi-Modell-Leitungen in Per-Unit. Wird vom JSON-Parser geladen. |
| `three_bus_poc.py` | **Programmatische Netzdefinition** desselben Netzes via `NetworkSpec`. Enthält zusätzlich `build_three_bus_case()` und `solve_three_bus_case()` als Hilfsfunktionen für Experimente und Jupyter-Notebooks. |
| `__init__.py` | Macht `cases/` als Python-Paket importierbar. |

---

## `tests/` — Automatisierte Tests

Alle Tests laufen via `pytest`. Konfiguration in `pyproject.toml` — kein
manueller `sys.path`-Eingriff nötig.

| Datei | Aufgabe |
|---|---|
| `conftest.py` | Gemeinsame Session-Fixtures: `three_bus_case` (lädt Netz aus JSON), `solved_three_bus` (gelöstes Netz). Session-Scope vermeidet Mehrfach-Kompilierung durch JAX. |
| `test_ybus.py` | Tests für Y-Bus-Aufbau (`core/ybus.py`) und den Netz-Compiler (`compile/network.py`): Topologiekorrektheit, Symmetrie, Vorzeichen der Diagonale, Handrechnung der Admittanzwerte. |
| `test_residuals.py` | Tests für Residuumsfunktionen (`core/residuals.py`): `state_to_voltage`, `calc_power_injection`, `power_flow_residual`, `residual_loss`. Prüft Form, Finite-Start-Endlichkeit, Null-Residuum im konvergierten Zustand. |
| `test_newton.py` | Tests für den Newton-Solver (`solver/newton.py`): Konvergenz, Ausgabe-Shapes, physikalische Plausibilität (Spannungen nahe 1 p.u., endliche Slack-Leistung), Gradient-Check via Finite Differences. |
| `test_io_parser.py` | Tests für JSON-Reader und Parser (`io/reader.py`, `io/parser.py`): Validierungsfehler (Duplikate, kein Slack, Null-Impedanz), korrekte p.u.-Konversion, nicht-zusammenhängende Bus-IDs, End-to-End-Roundtrip mit Solver. |
| `test_compile.py` | *Migriert nach `test_ybus.py`.* Kann gelöscht werden. |
| `test_numerics.py` | *Migriert nach `test_ybus.py` und `test_residuals.py`.* Kann gelöscht werden. |
| `test_solver.py` | *Migriert nach `test_newton.py`.* Kann gelöscht werden. |

---

## `experiments/` — Experimente *(geplant)*

Ein Skript pro Experiment, entsprechend dem geplanten Workflow:

| Datei | Aufgabe |
|---|---|
| `exp01_validate_pandapower.py` | Vergleich des JAX-Leistungsfluss-Ergebnisses mit pandapower als Referenzlöser. |
| `exp02_validate_gradients_fd.py` | Systematischer Gradient-Check: Autodiff-Gradienten vs. Finite-Differenzen für alle differenzierbaren Parameter. |
| `exp03_sensitivity_weather.py` | Sensitivitäten von Wettergrößen (Einstrahlung, Temperatur) auf Netzgrößen (Spannungen, Verluste) via Autodiff. |
| `exp04_model_swap.py` | Austausch des PV-Modells (physikalisch ↔ neuronales Netz) bei unverändertem Grid-Core. |
| `exp05_inverse_sizing_pv.py` | Inverse Optimierung: PV-Fläche A so wählen, dass eine Spannungsgrenze (z. B. 1,05 p.u.) nicht überschritten wird. |

---

## `notebooks/` — Exploration *(nicht für CI)*

| Datei | Aufgabe |
|---|---|
| `01_ybus_sanity.ipynb` | Interaktive Überprüfung der Y-Bus-Matrix: Stamping-Verfahren, Visualisierung der Admittanzmatrix. |
| `02_gradient_flow.ipynb` | Visualisierung des Gradientenflusses durch den differenzierbaren Leistungsfluss-Kern. |

---

## `docs/` — Dokumentation

| Datei | Aufgabe |
|---|---|
| `architektur.md` | Diese Datei. Beschreibt Aufgaben und Verantwortlichkeiten aller Projektdateien. |
| `theory.md` | Mathematische Herleitungen: Residuenformulierung, Jacobi-Berechnung in kartesischen Koordinaten (e,f-Darstellung), Herleitung der Gradienten. |
| `figures/` | Abbildungen für Dokumentation und Bachelorarbeit. |
