# diffpf

`diffpf` ist ein Proof of Concept fuer einen differenzierbaren stationaeren
AC-Power-Flow-Kern in JAX. Das Projekt ist im Rahmen einer Bachelorarbeit
entstanden und untersucht, wie elektrische Netzphysik als glatte Rechenschicht
in groessere Modellketten eingebettet werden kann.

Der zentrale Anwendungsfall ist nicht nur

```text
P/Q-Injektion -> AC-Power-Flow -> Knotenspannungen
```

sondern eine End-to-End-Kette wie

```text
Wetter -> PV-Modell -> P/Q-Einspeisung -> AC-Power-Flow -> elektrische Observables
```

Damit werden Sensitivitaeten nicht-elektrischer Groessen wie Einstrahlung,
Umgebungstemperatur oder Windgeschwindigkeit auf elektrische Zielgroessen
direkt per Automatic Differentiation berechenbar.

## Einordnung

`diffpf` ist ein wissenschaftlicher Demonstrator. Es ist kein vollstaendiger
Ersatz fuer `pandapower`, kein industrielles Netzberechnungsframework und kein
OPF-Tool. Der Wert des Projekts liegt in der nachvollziehbaren Kopplung von
stationaerer AC-Netzphysik, impliziter Differentiation und vorgelagerten
JAX-kompatiblen Modellen.

Die wichtigsten Leitfragen sind:

- Rechnet der JAX-Kern stationaere AC-Betriebspunkte konsistent?
- Stimmen lokale implizite Gradienten mit zentralen Finite Differences ueberein?
- Kann ein `pandapower`-naher Demonstrator in die JAX-Strukturen ueberfuehrt werden?
- Bleibt derselbe Power-Flow-Kern nutzbar, wenn verschiedene Upstream-Modelle
  P/Q-Injektionen liefern?

## Methodischer Kern

Die elektrische Physik wird intern in komplexer Form ausgewertet:

```text
I = Y_bus @ V
S = V * conj(I)
```

Der Solverzustand bleibt reell und nutzt rechteckige Spannungskoordinaten
`V = V_r + j V_i`. Fuer PQ-Busse werden P- und Q-Residuen geloest. Fuer
idealisierte PV-Busse ist im Kern eine Spannungsbetragsgleichung vorbereitet;
die validierten `pandapower`-Experimente verwenden jedoch bewusst den
scope-matched PQ-Modus ohne vollstaendige Generator-Spannungsregelung.

Fest gesetzte Designentscheidungen:

- durchgaengiges Per-Unit-System,
- statische Topologie und differenzierbare Parameter in `NetworkParams`,
- Newton-Raphson-Solve mit `jax.lax.while_loop`,
- Jacobi-Matrix im Solver via `jax.jacfwd`,
- Gradienten am konvergierten Betriebspunkt via `jax.lax.custom_root`,
- kein Unrolling der Newton-Iterationen fuer die Solverableitung,
- `float64` in JAX, aktiviert in `src/diffpf/__init__.py`.

## Datenpipeline

Die Software trennt bewusst Eingabe, statische Kompilierung und numerischen
Kern:

```text
pandapower net oder JSON
    -> RawNetwork / NetworkSpec
    -> compile_network(...)
    -> CompiledTopology + NetworkParams + PFState
    -> Newton / implicit solver
    -> Observables, Validierung, Experimente
```

`pandapower` bleibt Referenz- und Importwerkzeug. Der numerische Kern in
`core/`, `compile/` und `solver/` bleibt frei von `pandapower`-Abhaengigkeiten.

## Aktueller Modellumfang

Unterstuetzt sind im aktuellen Projektstand:

- stationaere symmetrische AC-Netze,
- genau ein Slack-Bus,
- PQ-Busse und eine idealisierte PV-Bus-Residualform im Kern,
- feste P/Q-Lasten und -Einspeisungen,
- Leitungen im Pi-Ersatzschaltbild,
- Zweiwicklungs-Transformatoren mit Tap und Phasenverschiebung,
- Shunts als konstante Admittanzen,
- einfache Switch-Vorverarbeitung,
- `pandapower`-Import fuer einen kontrollierten Teilumfang.

Die `pandapower`-Pipeline unterstuetzt insbesondere `bus`, `ext_grid`, `load`,
`sgen`, `gen`, `line`, `trafo`, `shunt` und einfache Switches. Geschlossene
Bus-Bus-Switches werden als Bus-Fusion behandelt; offene Line- oder
Trafo-Switches deaktivieren das jeweilige Element im unterstuetzten Scope.

Nicht Teil des aktuellen Scopes sind vollstaendige `pandapower`-Kompatibilitaet,
Controller, unsymmetrische oder dreiphasige Lastfluesse, Dreiwicklungs-
Transformatoren, `ward`/`xward`, `dcline`, detaillierte offene Leitungsenden,
Generator-Q-Limits und PV-PQ-Umschaltung.

## Demonstratoren

Der urspruengliche 3-Bus-Fall in `cases/three_bus_poc.py` und
`cases/three_bus_poc.json` bleibt als historischer Minimal- und Kontrollfall
erhalten:

- Bus 0: Slack / Umspannwerk,
- Bus 1: Last,
- Bus 2: PV-Einspeisung.

Der aktuelle Hauptdemonstrator ist `pandapower.networks.example_simple()`.
Dieses Netz enthaelt 110-kV- und 20-kV-Ebenen, einen Slack, einen
Zweiwicklungs-Transformator mit 150 Grad Phasenverschiebung, Leitungen,
Switches, eine Last, einen `gen`, einen `sgen` und einen Shunt.

Fuer die Upstream-Kopplung ist festgelegt:

```text
Kopplungsbus:       "MV Bus 2"
Ersetztes Element:  sgen "static generator"
Referenzwerte:      P = 2.0 MW, Q = -0.5 MVAr
Q/P-Verhaeltnis:    -0.25
Modellierung:       wetterabhaengige P/Q-Injektion, kein spannungsregelnder PV-Bus
```

Ab Experiment 3 wird ausschliesslich `example_simple()` verwendet. Das
statische `sgen` wird dort deaktiviert und durch JAX-kompatible Upstream-Modelle
ersetzt, ohne den Power-Flow-Kern zu veraendern.

## Experimente und Status

### Experiment 1: Forward-Validierung

Experiment 1 validiert den stationaeren Forward-Solve gegen `pandapower`.
Umgesetzt sind sowohl der urspruengliche 3-Bus-PoC als auch
`example_simple()`.

Fuer `example_simple()` gibt es zwei Modi:

- `scope_matched`: aktive `gen` werden zu `sgen(P, Q=0)` konvertiert; dieser
  Modus ist der strikte Vergleich.
- `original_pandapower`: `pandapower` nutzt den originalen PV-Bus-Generator;
  dieser Modus ist nur Kontextvergleich.

Alle dokumentierten Szenarien konvergieren. Im `scope_matched`-Modus liegen die
Knotenspannungen sehr nah an `pandapower` (`max |dV|` etwa `6e-5 pu`,
`max |dtheta|` etwa `0.0023 deg`). Ein bekannter systematischer Offset bleibt
bei Trafofluessen, Trafoverlusten und Slack-Leistung, vermutlich wegen nicht
vollstaendig identischer Trafo-Shift-/Verlustabbildung.

Wichtige Dateien:

- `experiments/exp01_validate_example_simple.py`
- `experiments/plot_exp01_validation_figures.py`
- `experiments/results/exp01_example_simple_validation/`

### Experiment 2: Gradientenvalidierung

Experiment 2 validiert implizite AD-Gradienten gegen zentrale Finite
Differences. Fuer `example_simple()` werden 48 Gradienten untersucht:

```text
3 Szenarien x 4 Eingangsparameter x 4 Observables
```

Alle 48 Gradienten sind gueltig; die absoluten AD-vs-FD-Fehler liegen im
Bereich von etwa `1e-9`, mit einem dokumentierten Maximum von etwa `4.7e-9`.
Groessere relative Fehler treten vor allem bei sehr kleinen
Shunt-Sensitivitaeten auf und sind wegen kleiner absoluter Fehler numerisch
unkritisch.

Wichtige Dateien:

- `experiments/exp02_validate_gradients_example_simple.py`
- `experiments/plot_exp02_gradient_figures.py`
- `experiments/results/exp02_example_simple_gradients/`

### Experiment 3: Cross-Domain PV Weather Sensitivity

Experiment 3 koppelt das `example_simple()`-Netz mit einem analytischen
PV-Wettermodell:

```text
g_poa_wm2, t_amb_c, wind_ms
    -> cell_temperature_noct_sam(...)
    -> pv_pq_injection_from_weather(...)
    -> P_pv, Q_pv am Bus "MV Bus 2"
    -> NetworkParams
    -> AC-Power-Flow
    -> elektrische Observables
```

Der aktuelle Lauf umfasst nach Erweiterung um einen Einstrahlungs-Sweep
108 Forward-Solves und 1296 Sensitivitaetszeilen. Ausgewertet werden unter
anderem `vm_mv_bus_2_pu`, `p_slack_mw`, `total_p_loss_mw` und
`p_trafo_hv_mw`. Ein kompakter AD-vs-FD-Spot-Check plausibilisiert die neue
Wetterkette.

Wichtige Dateien:

- `src/diffpf/models/pv.py`
- `experiments/exp03_cross_domain_pv_weather.py`
- `experiments/plot_exp03_figures.py`
- `experiments/results/exp03_cross_domain_pv_weather/`

### Experiment 4: Modulare Upstream-Kopplung

Experiment 4 zeigt, dass verschiedene vorgelagerte Modelle ueber dieselbe
P/Q-Schnittstelle an den unveraenderten Power-Flow-Kern gekoppelt werden
koennen. Verglichen werden:

- analytisches PV-Wettermodell,
- kleines JAX-MLP als P-only-Surrogat mit festem `Q = -0.25 * P`,
- direkte differenzierbare P/Q-Skalierungsbaseline.

Das MLP ist ein Distillation-Surrogat fuer den Modularitaetsnachweis, kein
Messdaten-Prognosemodell.

Wichtige Dateien:

- `src/diffpf/models/pq_surrogate.py`
- `experiments/exp04_modular_upstream_nn_surrogate.py`
- `experiments/results/exp04_modular_upstream_nn_surrogate/`

### Experiment 5: Screening und PV-Curtailment

Experiment 5 ist in zwei getrennte Schritte aufgeteilt.

Experiment 5a ist ein reduziertes Netzscreening auf `example_simple()`. Es
loest 48 PV-Screeningfaelle plus vier no-PV-Referenzen, berechnet
demonstratorinterne Stressindikatoren, selektiert Top-20-Betriebspunkte und
berechnet nur fuer diese lokale Sensitivitaeten. Zusaetzlich wird der separate
Sommer-Hoch-PV-Fall `selected_realistic_load0p4_g1200_t30` berechnet.

Experiment 5b optimiert fuer genau diesen ausgewaehlten Fall den
PV-Curtailment-Faktor `c in [0, 1]`. Die Optimierung nutzt eine
Sigmoid-Parametrisierung, einen glatten Export-Proxy `-p_slack_mw` und eine
1D-Grid-Referenz. Der Exportzielwert `7.0 MW` ist ein demonstratorinterner
Zielwert, keine normative Netzcode-Grenze.

Wichtige Dateien:

- `experiments/exp05a_network_screening.py`
- `experiments/exp05b_optimize_pv_curtailment.py`
- `experiments/results/exp05a_network_screening/`
- `experiments/results/exp05b_optimize_pv_curtailment/`

## Repository-Struktur

```text
src/diffpf/
  core/        Numerische Kernlogik: Typen, p.u., Y-Bus, Residuen, Observables
  compile/     NetworkSpec -> CompiledTopology + NetworkParams
  io/          JSON-Reader, Parser, pandapower-Adapter, Topologie-Helfer
  solver/      Newton-Raphson und implizit differenzierbarer Solver
  models/      PV-Modell und kleines P/Q-Surrogat
  validation/  pandapower-Referenz, Finite Differences, Gradient-Checks
  pipeline/    Platzhalter fuer End-to-End-Pipelines
  viz/         Platzhalter fuer Visualisierungshelfer

cases/         3-Bus-PoC als Python- und JSON-Fall
experiments/   Reproduzierbare Experiment- und Plot-Skripte
tests/         pytest-Suite fuer Kern, IO, Solver, Modelle und Artefakte
docs/          Kontext-, Architektur-, Modellierungs- und Experimentdokumente
```

## Installation

Das Projekt nutzt ein `src/`-Layout und kann direkt aus dem Repository
installiert werden.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Alternativ enthaelt `requirements.txt` eine manuelle Abhaengigkeitsliste fuer
direkte Installationen:

```powershell
python -m pip install -r requirements.txt
```

Die wichtigste Laufzeitbasis ist JAX auf CPU. `pandapower` wird fuer Import,
Referenzvalidierung und Tests verwendet, aber nicht im numerischen Kern.

## Quickstart

Minimaler 3-Bus-Solve:

```powershell
@'
from cases.three_bus_poc import solve_three_bus_case

result = solve_three_bus_case()
print("residual_norm:", float(result["residual_norm"]))
print("voltage_mag_pu:", result["voltage_mag_pu"])
'@ | python -
```

Tests:

```powershell
python -m pytest -q
```

Ausgewaehlte Experimente:

```powershell
python experiments/exp01_validate_example_simple.py
python experiments/exp02_validate_gradients_example_simple.py
python experiments/exp03_cross_domain_pv_weather.py
python experiments/exp04_modular_upstream_nn_surrogate.py
python experiments/exp05a_network_screening.py
python experiments/exp05b_optimize_pv_curtailment.py
```

Plot-Artefakte aus vorhandenen Ergebnisdateien:

```powershell
python experiments/plot_exp01_validation_figures.py
python experiments/plot_exp02_gradient_figures.py
python experiments/plot_exp03_figures.py
```

## Ergebnisartefakte

Die Experimente schreiben reproduzierbare CSV-/JSON-Artefakte unter
`experiments/results/`. Die wichtigsten Ergebnisordner sind:

- `exp01_example_simple_validation/`
- `exp02_example_simple_gradients/`
- `exp03_cross_domain_pv_weather/`
- `exp04_modular_upstream_nn_surrogate/`
- `exp05a_network_screening/`
- `exp05b_optimize_pv_curtailment/`

Die Plot-Skripte lesen vorhandene Artefakte und erzeugen Abbildungen, ohne neue
Power-Flow-Solves oder neue Gradientenlaeufe zu starten.

## Bekannte Grenzen

Die Validierung ist lokal und demonstratorbezogen. Sie ist kein mathematischer
Beweis fuer beliebige Netze oder Betriebspunkte.

Wichtige dokumentierte Grenzen:

- keine vollstaendige `pandapower`-Kompatibilitaet,
- keine Controller- oder Schutzlogik,
- keine Generator-Q-Limits,
- keine PV-PQ-Umschaltung,
- keine vollstaendige spannungsregelnde `pandapower.gen`-Semantik im
  scope-matched Validierungspfad,
- statische Topologie im JAX-Kern,
- keine Differentiation diskreter Schalt- oder Aktiv/Inaktiv-Entscheidungen,
- verbleibender Trafo-Offset im `example_simple()`-Vergleich,
- kein Nachweis fuer industrielle Groessennetze oder Echtzeitfaehigkeit.

Diese Grenzen blockieren die dokumentierte PV-Upstream-Kopplung nicht, weil sie
bewusst als glatte P/Q-Einspeisung an einem PQ-Bus modelliert wird.

## Dokumentation

Weitere Details liegen in `docs/`:

- `docs/context/project_context.md`: Projektziel, Scope und Demonstrator
- `docs/context/software_context.md`: Softwarearchitektur und Schichten
- `docs/context/modeling_assumptions.md`: Modellannahmen und Vorzeichen
- `docs/context/validation_status.md`: Validierungsstand
- `docs/context/known_limitations.md`: Grenzen und Future Work
- `docs/context/experiment_plan.md`: Experimentstrategie
- `docs/pandapower_io_pipeline.md`: pandapower-Importpipeline
- `docs/pandapower_example_simple_preparation.md`: Mapping von `example_simple()`
- `CHANGELOG.md`: chronologischer Entwicklungsstand

## Status in einem Satz

Der aktuelle Stand umfasst einen JAX-basierten, implizit differenzierbaren
stationaeren AC-Power-Flow-Kern, eine kontrollierte `pandapower`-I/O-Pipeline,
validierte Forward- und Gradientenergebnisse auf `example_simple()` sowie
umgesetzte Cross-Domain- und Modularitaetsexperimente mit PV-Upstream-Modellen.
