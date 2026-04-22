# Changelog

Dieses Dokument fasst die bisher umgesetzten Entwicklungsschritte im Projekt `diffpf`
zusammen, inklusive der Parser-Schicht und der bisher realisierten Experimente.

## 2026-04-22 - Refaktorierung: Physikalische Leitungsparameter im JSON-Eingabemodell

- `src/diffpf/io/reader.py` grundlegend überarbeitet.
- `RawLine` von p.u.-Direkteingabe (`r_pu`, `x_pu`, `b_shunt_pu`) auf physikalische Einheiten umgestellt.
- Zwei exklusiv wählbare Eingabeformen eingeführt:
  - **Form A (Direktwerte):** `r_ohm`, `x_ohm`, optional `b_shunt_s`
  - **Form B (Belagswerte):** `length_km` + `r_ohm_per_km` + `x_ohm_per_km` + optional `b_shunt_s_per_km` oder `c_nf_per_km`
- Mischformen und fehlende Pflichtfelder werden zur Ladezeit mit deutschen Fehlermeldungen abgewiesen.
- `f_hz` zu `RawBase` hinzugefügt; Pflichtfeld wenn `c_nf_per_km` verwendet wird (Kapazität → Suszeptanz: b = 2π·f·C).
- `src/diffpf/io/parser.py` erweitert.
- Private Zwischenform `_PhysicalLine(r_ohm, x_ohm, b_shunt_s)` als kanonische physikalische Repräsentation eingeführt.
- `_to_physical()` normiert beide Eingabeformen auf `_PhysicalLine`.
- `_physical_to_pu()` rechnet über `BaseValues` in Per-Unit um.
- Öffentliche Hilfsfunktion `line_to_pu(raw_line, base)` für externe Module ergänzt.
- `src/diffpf/core/units.py` erweitert.
- `BaseValues` um `f_hz`-Parameter, `y_base_s`-Attribut und `siemens_to_pu()`-Methode ergänzt.
- `src/diffpf/io/__init__.py` aktualisiert: `line_to_pu` öffentlich exportiert.
- `cases/three_bus_poc.json` auf physikalische Einheiten migriert.
- `f_hz: 50.0` in den `base`-Block aufgenommen.
- Leitungsparameter von p.u. auf Ohm / Siemens umgerechnet (Z_base = 0,16 Ω, Y_base = 6,25 S); numerische Äquivalenz zu den bisherigen Werten gewährleistet.
- `src/diffpf/validation/pandapower_ref.py` angepasst.
- Direkte `line.r_pu`-Zugriffe durch Aufrufe von `line_to_pu(line, base)` ersetzt.
- `BaseValues`-Konstruktoren um `f_hz`-Argument ergänzt.
- 13 neue Parser-Tests in `tests/test_io_parser.py` ergänzt:
  - Form-A-Parsing und exakte p.u.-Umrechnung
  - Shunt-Suszeptanz-Standardwert 0,0 bei fehlender Angabe
  - Form-B-Parsing mit `b_shunt_s_per_km` und `c_nf_per_km`
  - Fehlerfall: `c_nf_per_km` ohne `f_hz`
  - Fehlerfall: Mischform A+B
  - Fehlerfall: keine gültige Form
  - Fehlerfall: beide Shunt-Spezifikationen gleichzeitig
  - Fehlerfall: Null-Impedanz nach Belagsexpansion
  - `f_hz`-Validierung in `BaseValues`

## 2026-04-22 - Experiment 1: Persistenz der Validierungsergebnisse

- `experiments/exp01_validate_pandapower.py` um CSV/JSON-Export erweitert.
- Zwei flache Zeilentypen als Dataclasses eingeführt:
  - `_SummaryRow` – ein Eintrag pro Betriebspunkt (Konvergenz, Iterationen, Residualnorm, Slack-Leistungen, Verluste, alle Abweichungsmetriken)
  - `_LineFlowRow` – ein Eintrag pro (Betriebspunkt, Leitung) mit JAX- und pandapower-Flüssen sowie absoluten Differenzen
- Exportierte Artefakte:
  - `experiments/results/exp01_pandapower_validation/validation_summary.csv`
  - `experiments/results/exp01_pandapower_validation/validation_summary.json`
  - `experiments/results/exp01_pandapower_validation/line_flows.csv`
  - `experiments/results/exp01_pandapower_validation/line_flows.json`

## 2026-04-13 - Experiment 2: Gradientenvalidierung

- `src/diffpf/solver/implicit.py` ergänzt.
- Fachliche API `solve_power_flow_implicit()` eingeführt.
- `jax.lax.custom_root` für implizite Differenzierung des stationären Power-Flow-Root-Problems verwendet.
- Vorwärtslösung des impliziten Solvers bewusst über den bestehenden Newton-Solver geführt, damit keine zweite physikalische Formulierung entsteht.
- `solve_power_flow_implicit_result()` als diagnostische Variante mit Residualnorm und Loss ergänzt.
- `NewtonResult` in `src/diffpf/solver/newton.py` JIT-kompatibler gemacht, indem `iterations` und `converged` als JAX-Werte erhalten bleiben.
- `src/diffpf/core/observables.py` ergänzt.
- Solver-unabhängige Observables eingeführt:
  - Spannungsbeträge der Nicht-Slack-Busse
  - Spannungswinkel der Nicht-Slack-Busse
  - Slack-Wirkleistung
  - Slack-Blindleistung
  - Gesamtwirkverluste
  - Gesamtblindverluste
  - gerichtete Leitungswirk- und Blindleistungsflüsse
  - Leitungsverluste
- `src/diffpf/validation/gradient_check.py` ergänzt.
- Wiederverwendbare Gradient-Check-Helfer eingeführt:
  - Mapping fachlicher Eingänge `P_load`, `Q_load`, `P_pv`, `Q_pv` auf `NetworkParams`
  - Mapping fachlicher Outputs auf skalare Observables
  - AD-vs-FD-Vergleich mit robustem relativen Fehler
  - aggregierte Fehlerstatistik pro Betriebspunkt
  - kleine FD-Schrittweitenstudie
- `experiments/exp02_validate_gradients.py` ergänzt.
- Experiment 2 validiert lokale implizite Gradienten gegen zentrale Finite Differences.
- Wiederverwendete Betriebspunkte:
  - `low_pv`
  - `medium_pv`
  - `high_pv`
- Exportierte Artefakte:
  - `experiments/results/exp02_gradient_validation/gradient_table.csv`
  - `experiments/results/exp02_gradient_validation/gradient_table.json`
  - `experiments/results/exp02_gradient_validation/error_summary.csv`
  - `experiments/results/exp02_gradient_validation/error_summary.json`
  - `experiments/results/exp02_gradient_validation/fd_step_study.csv`
  - `experiments/results/exp02_gradient_validation/fd_step_study.json`
- Neue Tests ergänzt:
  - `tests/test_implicit_solver_matches_newton.py`
  - `tests/test_implicit_gradients_vs_fd.py`
  - `tests/test_observables.py`

## 2026-04-10 - Experiment 1: Validierung gegen pandapower

- `src/diffpf/validation/pandapower_ref.py` ergänzt.
- Referenzadapter für `pandapower` implementiert.
- Mehrere Betriebspunkte für das 3-Bus-PoC-Netz eingeführt:
  - `low_pv`
  - `medium_pv`
  - `high_pv`
- Vergleichsmetriken implementiert:
  - Konvergenz
  - Iterationszahl
  - Residualnorm
  - Knotenspannungsbeträge
  - Spannungswinkel
  - Gesamtverluste
  - Leitungsflüsse
- `experiments/exp01_validate_pandapower.py` ergänzt.
- Konsolenbericht für Experiment 1 implementiert.
- `tests/test_pandapower_validation.py` ergänzt.
- Ergebnis: Der JAX-Power-Flow-Kern stimmt im 3-Bus-PoC-Netz innerhalb numerischen Rundungsrauschens mit `pandapower` überein.

## 2026-04-10 - Numerischer V1-Kern

- Grundlegende V1-Architektur umgesetzt.
- `src/diffpf/core/types.py` als zentrale Datentyp-Schicht etabliert.
- Wichtige Datentypen:
  - `BusSpec`
  - `LineSpec`
  - `NetworkSpec`
  - `CompiledTopology`
  - `NetworkParams`
  - `PFState`
- JAX-Pytree-Registrierung mit `meta_fields` und `data_fields` verwendet, um statische Topologie von differenzierbaren Parametern zu trennen.
- `src/diffpf/core/units.py` eingeführt.
- `BaseValues` für Per-Unit-Konvertierungen implementiert.
- `src/diffpf/core/ybus.py` eingeführt.
- `build_ybus()` als Pi-Modell-Stempelverfahren implementiert.
- `src/diffpf/core/residuals.py` eingeführt.
- Kernfunktionen implementiert:
  - `state_to_voltage()`
  - `calc_power_injection()`
  - `power_flow_residual()`
  - `residual_loss()`
- `src/diffpf/compile/network.py` eingeführt.
- `compile_network()` überführt menschenfreundliche Netzspezifikationen in JAX-Arrays.
- `src/diffpf/solver/newton.py` eingeführt.
- Newton-Raphson-Solver mit `jax.lax.while_loop` und `jax.jacfwd` implementiert.
- `src/diffpf/validation/finite_diff.py` eingeführt.
- `central_difference()` als einfacher zentraler Finite-Difference-Helfer ergänzt.
- `cases/three_bus_poc.py` und `cases/three_bus_poc.json` als 3-Bus-Demonstrator etabliert.
- Erste Tests für Compiler, Y-Bus, Residuen, Newton-Solver und Gradienten-Smoke-Checks ergänzt.

## Parser-Schicht

- `src/diffpf/io/reader.py` ergänzt.
- Rohdatenmodell für JSON-Netze implementiert:
  - `RawBase`
  - `RawBus`
  - `RawLine`
  - `RawNetwork`
- JSON-Loader `load_json()` implementiert.
- Semantische Validierung vor JAX-Kontakt eingeführt:
  - genau ein Slack-Bus
  - keine doppelten Bus-IDs
  - keine doppelten Leitungs-IDs
  - gültige Leitungsendpunkte
  - keine Self-Loops
  - keine Null-Impedanzen
  - positive Basisgrößen
- `src/diffpf/io/parser.py` ergänzt.
- Parser `parse()` implementiert.
- Parser-Aufgaben:
  - physikalische Größen über `BaseValues` in Per-Unit umrechnen
  - externe Bus-IDs auf interne 0-basierte Indizes abbilden
  - Slack-Spannung von Polar- in Rechteckkoordinaten umrechnen
  - `NetworkSpec` aufbauen
  - `compile_network()` aufrufen
  - Flat-Start-`PFState` erzeugen
- Convenience-Funktion `load_network(path)` eingeführt.
- Parser-Tests in `tests/test_io_parser.py` ergänzt.

## Dokumentation und Statusdateien

- `docs/architektur.md` enthält eine Datei- und Architekturübersicht.
- `docs/software_status.txt` enthält einen kompakten Projekt-Snapshot für weitere Chatbot- oder Planungsdialoge.
- `docs/CHANGELOG.md` wurde als fortlaufender Änderungslog ergänzt.

