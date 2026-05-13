# Architektur und Dateiübersicht

Dieses Dokument beschreibt den aktuellen Aufbau von `diffpf`, die
Verantwortlichkeiten der wichtigsten Dateien und die beabsichtigte
Abhängigkeitsstruktur des Projekts.

---

## Wurzelverzeichnis

| Datei / Ordner | Aufgabe |
|---|---|
| `pyproject.toml` | Projektmetadaten, Runtime- und Dev-Dependencies, pytest- sowie Formatierungs-Konfiguration. |
| `requirements.txt` | Alternative manuelle Abhängigkeitsliste für direkte Installationen. |
| `.gitignore` | Git-Ignore-Regeln für virtuelle Environments, Caches und Build-Artefakte. |
| `README.md` | Kurzüberblick, Motivation und Einstieg in das Projekt. |
| `AGENTS.md` | Projektkontext und Guardrails für Coding-Agenten. |
| `cases/` | Kanonische Netzdefinitionen als Daten. |
| `experiments/` | Reproduzierbare Skripte für die wissenschaftlichen Experimente. |
| `tests/` | pytest-Suite, die die Kernlogik und Validierung absichert. |
| `docs/` | Architektur-, Status-, Changelog- und Visualisierungsdokumente. |

---

## Zielarchitektur

Das Projekt trennt bewusst drei Ebenen:

1. **Menschenfreundliche Eingabeebene**
   JSON-Dateien und Raw-Dataclasses mit physikalischen Einheiten.

2. **Interne JAX-Netzdarstellung**
   `CompiledTopology`, `NetworkParams`, `PFState`.

3. **Numerisch-differenzierbarer Kern**
   Y-Bus, Residuen, Newton-Solver, implizite Differenzierung und fachliche Observables.

Wichtige Abhängigkeitsrichtung:

```text
io -> compile -> core -> solver
                     -> validation
core/solver -> pipeline -> experiments
validation darf Referenzcode importieren, wird aber von nichts anderem benötigt.
```

Methodische Invarianten:

- rechteckige Spannungsdarstellung `V = Vr + j Vi`
- intern komplexe AC-Physik, aber reeller Solverzustand
- konsequentes p.u.-System
- JIT-kompatible, funktionale Implementierung
- statische Topologie, differenzierbare Parameter und Zustände
- `pandapower` nur in `validation/` und Tests

---

## `src/diffpf/` – Python-Paket

### `src/diffpf/__init__.py`

Paket-Einstiegspunkt. Aktiviert `jax_enable_x64`, damit der Kern konsistent in
`float64` rechnet.

---

## `src/diffpf/core/` – Reiner numerischer Kern

Dieses Modul enthält die physikalische und numerische Kernlogik. Es importiert
nicht aus `io/` und enthält keine JSON- oder Referenzsolver-Logik.

| Datei | Aufgabe |
|---|---|
| `__init__.py` | Re-exportiert die wichtigsten Typen und Kernfunktionen aus `types`, `units`, `ybus`, `residuals` und `observables`. |
| `types.py` | Definiert die zentralen Datentypen. `BusSpec`, `LineSpec`, `NetworkSpec` sind menschenfreundliche Spezifikationen. `CompiledTopology` enthält statische Indexstrukturen. `NetworkParams` enthält differenzierbare physikalische Parameter. `PFState` repräsentiert den rechteckigen Spannungszustand der Nicht-Slack-Busse. |
| `units.py` | Enthält `BaseValues` für p.u.-Konvertierungen zwischen MW, MVAR, kV, Ohm, Siemens und p.u. |
| `ybus.py` | Implementiert `build_ybus()` als Stamping-Verfahren für Pi-Leitungen. |
| `residuals.py` | Implementiert `state_to_voltage()`, `calc_power_injection()`, `power_flow_residual()` und `residual_loss()`. Dies ist die eigentliche stationäre Power-Flow-Formulierung. |
| `observables.py` | Implementiert `power_flow_observables()` und die Dataclass `PowerFlowObservables`. Diese solver-unabhängige Auswertung liefert Spannungsbeträge, Winkel, Slack-Leistung, Gesamtverluste und Leitungsflüsse aus einem gelösten Zustand. |

### Kerngedanke

`core/` ist die stabile Physikschicht. Solver und Experimente bauen auf dieser
Schicht auf, ohne die mathematische Formulierung zu duplizieren.

---

## `src/diffpf/io/` – Eingabemodell, Reader und Parser

Diese Schicht ist die einzige Brücke zwischen menschenfreundlicher Eingabe und
der internen JAX-Welt.

| Datei | Aufgabe |
|---|---|
| `__init__.py` | Re-exportiert `load_json`, `load_network`, `parse`, `line_to_pu`, `RawNetwork`. |
| `reader.py` | Liest Netz-JSON-Dateien ein und erzeugt `RawBase`, `RawBus`, `RawLine`, `RawNetwork`. Validiert Topologie, Bus-Typen, Leitungsdefinitionen, Basisgrößen und die beiden Leitungsformen: Form A (Gesamtwerte) und Form B (Beläge × Länge, optional mit `c_nf_per_km`). |
| `parser.py` | Normalisiert Leitungen auf physikalische Gesamtwerte, rechnet sie mit `BaseValues` in p.u. um, mappt externe Bus-IDs auf interne Indizes, zerlegt den Slack-Setpoint in Rechteckkoordinaten und ruft `compile_network()` auf. Zusätzlich erzeugt es einen Flat-Start-`PFState`. |

### Reader- und Parser-Konzept

- `reader.py` bleibt reines Python ohne JAX-Arrays.
- `parser.py` ist die kontrollierte Konversionsschicht in die interne
  Netzdarstellung.
- `line_to_pu()` ist eine Convenience-Funktion für andere Schichten, die
  einzelne Leitungsdefinitionen in p.u. benötigen, etwa die Referenzvalidierung.

---

## `src/diffpf/compile/` – Interner Netz-Compiler

| Datei | Aufgabe |
|---|---|
| `__init__.py` | Re-exportiert `compile_network`. |
| `network.py` | Wandelt ein `NetworkSpec` in `(CompiledTopology, NetworkParams)` um. Berechnet Serienadmittanzen aus `r_pu` und `x_pu`, prüft Konsistenzbedingungen und baut JAX-Arrays in `float64` bzw. `int32`. |

### Ergebnis des Compilers

- `CompiledTopology` = statische Struktur für JIT und Indexing
- `NetworkParams` = differenzierbare physikalische Parameter

---

## `src/diffpf/solver/` – Vorwärtssolve und implizite Differenzierung

| Datei | Aufgabe |
|---|---|
| `__init__.py` | Re-exportiert `NewtonOptions`, `NewtonResult`, `ImplicitPowerFlowResult`, `solve_power_flow()`, `solve_power_flow_result()`, `solve_power_flow_implicit()` und `solve_power_flow_implicit_result()`. |
| `newton.py` | Implementiert den gedämpften Newton-Raphson-Solver mit `jax.lax.while_loop`. Die Jacobi-Matrix wird via `jax.jacfwd` gebildet. `NewtonResult` kapselt Lösung, Residualnorm, Loss, Iterationszahl und Konvergenzstatus. |
| `implicit.py` | Implementiert den implizit differenzierbaren Solver mittels `jax.lax.custom_root`. Die Vorwärtslösung nutzt weiterhin den bestehenden Newton-Solver, die Rückwärtsableitung erfolgt über den linearisierten Root-Solve am Konvergenzpunkt. |

### Designentscheidung

Es gibt keine zweite physikalische Solverformulierung. `implicit.py` verwendet
denselben Residuenkern wie `newton.py`, ersetzt aber die Gradientenableitung
durch implizite Differenzierung.

---

## `src/diffpf/validation/` – Referenz- und Gradientenvalidierung

Hier liegt Validierungslogik, die bewusst außerhalb von `core/` und `solver/`
bleibt.

| Datei | Aufgabe |
|---|---|
| `__init__.py` | Re-exportiert Finite-Difference-Helfer, Szenarien und Validierungsfunktionen aus `finite_diff.py`, `pandapower_ref.py` und `gradient_check.py`. |
| `finite_diff.py` | Enthält `central_difference()` für zentrale finite Differenzen. |
| `pandapower_ref.py` | Referenzadapter für Experiment 1. Baut Betriebspunkte auf Basis von `RawNetwork`, löst sie mit JAX und `pandapower` und vergleicht Spannungen, Winkel, Verluste und Leitungsflüsse. |
| `gradient_check.py` | Hilfsfunktionen für Experiment 2. Mappt fachliche Eingänge wie `P_load`, `Q_load`, `P_pv`, `Q_pv` auf `NetworkParams`, berechnet Observables, vergleicht AD-Gradienten gegen FD und erzeugt Fehlerstatistiken und Schrittweitenstudien. |

### Rolle von `validation/`

Diese Schicht dient der wissenschaftlichen Absicherung des Kerns. Sie ist nicht
Teil des produktionsartigen numerischen Hot Paths.

---

## `src/diffpf/models/` – Kompatibilitätsschicht

| Datei | Aufgabe |
|---|---|
| `__init__.py` | Re-exportiert Typen aus `diffpf.core.types`. |
| `network.py` | Kompatibilitätsshims für ältere Imports. |

Diese Schicht ist vor allem für Rückwärtskompatibilität vorhanden und soll
langfristig nicht zur primären API werden.

---

## `src/diffpf/numerics/` – Kompatibilitätsschicht

| Datei | Aufgabe |
|---|---|
| `__init__.py` | Re-exportiert numerische Funktionen aus `diffpf.core`. |
| `power_flow.py` | Kompatibilitätsshims für ältere Importe der numerischen Kernfunktionen. |

---

## `src/diffpf/pipeline/` – Platzhalter für V2+

| Datei | Aufgabe |
|---|---|
| `__init__.py` | Platzhalter für spätere End-to-End-Pipelines wie Wetter -> PV-Modell -> Power Flow -> Netzgrößen. |

---

## `src/diffpf/viz/` – Platzhalter für Visualisierung

| Datei | Aufgabe |
|---|---|
| `__init__.py` | Platzhalter für spätere Plot-Helfer und Berichtsgrafiken. |

---

## `cases/` – Netzdefinitionen

| Datei | Aufgabe |
|---|---|
| `three_bus_poc.json` | Kanonische JSON-Netzdefinition des 3-Bus-PoC inklusive Basisgrößen, Busse und Leitungen. Unterstützt die aktuell implementierten Reader-/Parser-Funktionen. |
| `three_bus_poc.py` | Programmatische Netzdefinition und kleine Hilfsfunktionen für manuelle Experimente oder Notebooks. |
| `__init__.py` | Macht `cases/` importierbar. |

---

## `experiments/` – Reproduzierbare wissenschaftliche Experimente

| Datei | Aufgabe |
|---|---|
| `exp01_validate_pandapower.py` | Experiment 1: Vorwärtsvalidierung des AC-Power-Flow-Kerns gegen `pandapower`. Exportiert Ergebnisübersichten und Leitungsflussdateien nach `experiments/results/exp01_pandapower_validation/`. |
| `exp02_validate_gradients.py` | Experiment 2: Gradientenvalidierung des implizit differenzierbaren Solvers gegen zentrale finite Differenzen. Exportiert Gradiententabellen, Fehlersummen und eine Schrittweitenstudie nach `experiments/results/exp02_gradient_validation/`. |
| `results/` | Artefaktordner der Experimente mit CSV- und JSON-Exports für Bericht, Plots und weitere Auswertung. |

### Status

Experiment 1 und Experiment 2 sind implementiert und dienen als Referenz für
die aktuelle Kernarchitektur.

---

## `tests/` – Automatisierte Tests

Die Testsuite spiegelt die aktuelle Architektur wider und prüft Forward-Solve,
Parser, Observables, Referenzvergleich und Gradientenvalidierung.

| Datei | Aufgabe |
|---|---|
| `conftest.py` | Gemeinsame Fixtures für das 3-Bus-Netz und gelöste Standardfälle. |
| `test_ybus.py` | Tests für Compiler und Y-Bus-Aufbau. |
| `test_residuals.py` | Tests für die Residuenfunktionen und die stationäre Formulierung. |
| `test_newton.py` | Tests für den Newton-Solver und grundlegende Gradientensmokes. |
| `test_io_parser.py` | Tests für Reader, Parser, Leitungsformen und JSON->JAX-Roundtrip. |
| `test_pandapower_validation.py` | Regressionstest für Experiment 1 gegen `pandapower`. |
| `test_observables.py` | Tests für solver-unabhängige Observables. |
| `test_implicit_solver_matches_newton.py` | Tests, dass impliziter Solver und Newton-Solver denselben Vorwärtszustand liefern. |
| `test_implicit_gradients_vs_fd.py` | Tests für implizite Gradienten gegen zentrale finite Differenzen über mehrere Szenarien. |

---

## `docs/` – Dokumentation

| Datei | Aufgabe |
|---|---|
| `architektur.md` | Diese Datei. Aktueller Überblick über Struktur und Verantwortlichkeiten. |
| `software_status.txt` | Kompakter Softwarestand als Text-Snapshot. |
| `ki_softwareueberblick.txt` | Stark verdichtete Projektübersicht für andere KI-Modelle. |
| `datenfluss_visualisierung.md` | Visualisierung des Datenflusses beim Einlesen und beim Lösen/Differenzieren. |
| `CHANGELOG.md` | Chronologische Zusammenfassung der bisher umgesetzten Schritte. |
| `figures/` | Ablageort für spätere Berichtsgrafiken und Abbildungen. |

---

## Aktueller Gesamtstatus

Der aktuelle Stand deckt die ersten drei Phasen des Projekts in funktionaler
Form ab:

- Kern-Datentypen, Y-Bus und Residuen sind implementiert.
- Newton-Raphson löst den stationären Power Flow.
- Der Vorwärtssolve ist gegen `pandapower` validiert.
- Implizite Differenzierung via `custom_root` ist implementiert.
- Gradienten sind für das 3-Bus-PoC gegen zentrale finite Differenzen validiert.

Noch nicht umgesetzt sind die Upstream-Modelle und Pipelineschichten wie
`pv_physical`, `pv_nn`, `weather_to_pf` oder inverse Optimierung.
