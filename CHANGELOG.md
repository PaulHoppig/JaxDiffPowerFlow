# Changelog

## 2026-05-01 - Analytisches PV-Upstream-Modell fuer example_simple()

### Geaenderte Dateien
- `src/diffpf/models/pv.py` - analytisches, JAX-kompatibles PV-P/Q-Modell mit
  `alpha`, explizitem `kappa = Q/P`, `PVInjection`-Rueckgabeobjekt und
  Adapterhelfern fuer Bus-Injektionen in `NetworkParams`
- `experiments/check_pv_coupling_baseline.py` - Baseline-Kopplung nutzt das
  neue gemeinsame PV-P/Q-Interface
- `tests/test_pv_model.py` - erweitert um Referenzpunkt, Einstrahlungs- und
  Temperaturverhalten, Q/P-Verhaeltnis, Autodiff und den
  `example_simple()`-Kopplungspunkt
- `src/diffpf/models/__init__.py` - neue PV-Modell-API exportiert

### Modellannahmen
- Die PV-Anlage ersetzt weiterhin das `sgen "static generator"` am Bus
  `"MV Bus 2"` fachlich als wetterabhaengige PQ-Einspeisung.
- Der Bus bleibt ein PQ-Bus; es gibt keine echte PV-Bus-Spannungsregelung,
  keine Q-Limits, keine PV-PQ-Umschaltung und keine Controllerlogik.
- Das Basismodell verwendet bewusst `Q_pv = kappa * P_pv` statt einer
  cos(phi)-Parametrisierung.
- Es wird keine Sattigung oder harte/glatte Begrenzung angewendet, damit der
  Referenzpunkt exakt und leicht testbar reproduziert wird.

## 2026-04-30 - PV-Kopplungspunkt example_simple()

### Neue Dateien
- `src/diffpf/models/pv.py` - JAX-kompatibles PV-P/Q-Kopplungsinterface mit
  zentralen Konstanten fuer den `example_simple()`-Kopplungspunkt
- `experiments/check_pv_coupling_baseline.py` - kleiner
  Baseline-Reproduktionscheck fuer die Kopplung ohne neues Experiment
- `tests/test_pv_model.py` - Unit-Tests fuer PV-Leistung, Q/P-Verhaeltnis,
  Bus-Injektion und Autodiff

### Festlegung
- Kopplungspunkt: Bus `"MV Bus 2"`
- Ersetztes Element: sgen `"static generator"`
- Referenzwerte: `P = 2.0 MW`, `Q = -0.5 MVAr`, `Q/P = -0.25`
- Der Bus bleibt ein PQ-Bus; die PV-Anlage wird als wetterabhaengige
  P/Q-Einspeisung modelliert.

### Validierung
Der Baseline-Check entfernt das vorhandene sgen, injiziert bei
`G = 1000 W/m^2` und `T_cell = 25 degC` dieselben P/Q-Werte ueber das neue
JAX-Interface und vergleicht Busspannung, Slack P/Q sowie Wirkleistungsverluste.

## 2026-04-29 - Experiment 2b: Gradientenvalidierung example_simple()

### Neue Dateien
- `experiments/exp02_validate_gradients_example_simple.py` - kompakte
  AD-vs-Finite-Differences-Validierung fuer das scope-matched pandapower
  `example_simple()`-Netz
- `tests/test_exp02_example_simple_gradients_outputs.py` - Schema- und
  Exporttests fuer das neue Experiment ohne schweren Solverlauf

### Szenarien
`base`, `load_high`, `sgen_high`

### Untersuchte Eingangsparameter
- `load_scale_mv_bus_2` - skaliert P und Q der vorhandenen Last an MV Bus 2
- `sgen_scale_static_generator` - skaliert P und Q des vorhandenen statischen
  Generators
- `shunt_q_scale` - skaliert die vorhandene Shunt-Suszeptanz
- `trafo_x_scale` - skaliert die Serienreaktanz des vorhandenen
  Zweiwicklungs-Transformators

### Untersuchte Ausgangsobservables
- `vm_mv_bus_2_pu`
- `p_slack_mw`
- `total_p_loss_mw`
- `p_trafo_hv_mw`

### Exportierte Artefakte
Alle Dateien werden unter
`experiments/results/exp02_example_simple_gradients/` geschrieben:
`gradient_table.csv/json`, `error_summary.csv/json`,
`fd_step_study.csv/json`, `metadata.json`, `README.md`.

### Bewusste Begrenzung
Der Pflichtlauf umfasst genau 3 Szenarien x 4 Eingangsparameter x 4
Ausgangsobservables = 48 Gradienten. Die FD-Schrittweitenstudie ist auf drei
repraesentative Gradienten beschraenkt.

### Bekannte Einschraenkungen
- Das aktive `gen` wird im scope-matched Modell als `sgen(P, Q=0)` behandelt.
- Keine echte PV-Bus-Spannungsregelung, keine Q-Limits, keine PV-PQ-Umschaltung.
- Keine Controllerlogik und keine vollstaendige pandapower-Generatorsemantik.
- Die Topologie bleibt konstant; nur kontinuierliche Skalierungsparameter
  werden untersucht.

## 2026-04-29 - Experiment 1b: Validierung example_simple()

### Neue Dateien
- `experiments/exp01_validate_example_simple.py` – Vollständige Validierung des
  pandapower `example_simple()`-Netzes gegen diffpf; direkt ausführbar
- `tests/test_exp01_example_simple_outputs.py` – 28 Tests für Experiment 1b
  (Importierbarkeit, Spalten, Physik, CSV/JSON-Export); 1 Slow-Integrationstest

### Szenarien
`base`, `load_low`, `load_high`, `sgen_low`, `sgen_high`,
`combined_high_load_low_sgen`, `combined_low_load_high_sgen`

### Zwei Referenzmodi
- **`scope_matched`** – `gen` wird in `sgen(P, Q=0)` umgewandelt; pandapower und
  diffpf verwenden dasselbe Modell; strikte numerische Validierung möglich
- **`original_pandapower`** – Original-Netz mit PV-Bus in pandapower; diffpf
  weiterhin mit `sgen(Q=0)`; Kontextvergleich (Q-Abweichungen erwartet)

### Exportierte Artefakte (in `experiments/results/exp01_example_simple_validation/`)
`validation_summary.csv/json`, `bus_results.csv/json`, `slack_results.csv/json`,
`line_flows.csv/json`, `trafo_flows.csv/json`, `losses.csv/json`,
`structure_summary.csv/json`, `metadata.json`, `README.md`

### Initialisierungsstrategie
Trafo-shift-aware Start: HV-Busse am Slack-Winkel, LV-Busse bei
`slack_angle − trafo_shift_deg`; vermeidet Flat-Start-Divergenz bei 150°-Trafo

### Bekannte Einschränkungen
- gen ohne Spannungsregelung (PV-Bus nur in pandapower, nicht in diffpf)
- Kein Q-Limit-Enforcement
- Keine PV↔PQ-Umschaltung
- `original_pandapower`-Modus kein strikter Gleichheitstest



Dieses Dokument fasst die bisher umgesetzten Entwicklungsschritte im Projekt `diffpf`
zusammen, inklusive der Parser-Schicht und der bisher realisierten Experimente.

## 2026-04-28 - pandapower I/O-Pipeline

### Neue Dateien
- `src/diffpf/io/pandapower_adapter.py` – pandapower-Netzobjekt → NetworkSpec
- `src/diffpf/io/topology_utils.py` – `merge_buses()` via Union-Find für Bus-Bus-Switch-Fusion
- `docs/pandapower_io_pipeline.md` – eigenständige Dokumentation der Pipeline
- `docs/pandapower_example_simple_preparation.md` – Netzaufbau und Mapping für example_simple()
- `tests/test_pandapower_adapter.py` – 16 Tests (15 passed, 1 xfail) für die pandapower-Pipeline
- `tests/test_json_trafo_shunt.py` – 11 Tests für das erweiterte JSON-Format

### Geänderte Dateien
- `src/diffpf/core/types.py` – `TrafoSpec`, `ShuntSpec` hinzugefügt; `NetworkSpec` um `trafos`, `shunts`, `v_set_pu` erweitert; `NetworkParams` um Trafo- und Shunt-Arrays erweitert (mit Defaults)
- `src/diffpf/core/ybus.py` – Trafo-Pi-Modell (off-nominal tap + Phasenverschiebung) und Shunt-Diagonal-Stamp ergänzt
- `src/diffpf/compile/network.py` – Kompilierung von TrafoSpec und ShuntSpec hinzugefügt
- `src/diffpf/io/reader.py` – `RawTrafo`, `RawShunt` Dataclasses; `RawNetwork` um `trafos`/`shunts` erweitert; Validierungsregeln ergänzt
- `src/diffpf/io/parser.py` – `_raw_trafo_to_spec()`, `_raw_shunt_to_spec()` hinzugefügt; `_build_spec()` erweitert
- `src/diffpf/io/__init__.py` – `from_pandapower`, `load_pandapower_json`, `RawTrafo`, `RawShunt` exportiert
- `src/diffpf/core/__init__.py` – `TrafoSpec`, `ShuntSpec` exportiert

### Neue Funktionen / Klassen
- `from_pandapower(net) -> NetworkSpec`
- `load_pandapower_json(path) -> NetworkSpec`
- `merge_buses(bus_ids, switch_pairs) -> dict[int, int]`
- `TrafoSpec` (Dataclass, frozen)
- `ShuntSpec` (Dataclass, frozen)
- `RawTrafo` (Dataclass)
- `RawShunt` (Dataclass)
- `_raw_trafo_to_spec(raw, base, id_to_idx) -> TrafoSpec`
- `_raw_shunt_to_spec(raw, base, id_to_idx) -> ShuntSpec`

### Unterstützte pandapower-Elemente
ext_grid (Slack), bus (nach Switch-Fusion), load, sgen, gen (PV, kein Q-Enforcement),
line (Pi-Modell), trafo (2-Wicklungs-Pi-Modell mit tap/shift), shunt

### Bekannte Einschränkungen
- PV-Busse werden wie PQ-Busse behandelt (kein Spannungsregler)
- Flat-Start versagt bei Netzen mit >90° Trafo-Phasenverschiebung
- trafo3w, xward, ward, impedance, dcline werden nicht unterstützt

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

