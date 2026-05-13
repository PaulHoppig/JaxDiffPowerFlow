# Datenfluss-Visualisierung

Dieses Dokument visualisiert den aktuellen Datenfluss in `diffpf` in zwei
Schritten:

1. Einlesen und Konvertieren des Eingabenetzes in die interne JAX-Netzdarstellung
2. Datenfluss von der internen JAX-Netzdarstellung durch numerischen Kern,
   Solver und implizite Differenzierung

Die Darstellung orientiert sich an der aktuell implementierten Struktur in
`io/`, `compile/`, `core/`, `solver/` sowie an den bereits vorhandenen
Experimenten 1 und 2.

## 1. Eingabenetz -> interne Netzdarstellung

```mermaid
flowchart TD
    A["Netzdatei<br/>cases/three_bus_poc.json"] --> B["io.reader.load_json(path)"]
    B --> C["JSON -> RawBase / RawBus / RawLine / RawNetwork"]
    C --> D["Semantische Validierung in reader.py<br/>- genau ein Slack-Bus<br/>- gueltige Leitungsform A/B<br/>- keine Nullimpedanz<br/>- gueltige Endpunkte<br/>- positive Basisgroessen"]
    D --> E["RawNetwork<br/>(reines Python, sortiert nach id)"]

    E --> F["io.parser.parse(raw)"]
    F --> G["BaseValues(s_mva, v_kv, f_hz)"]
    E --> H["Bus-ID-Mapping<br/>externe IDs -> interne 0-basierte Indizes"]
    E --> I["Leitungsnormalisierung _to_physical()<br/>Form A: Gesamtwerte in Ohm / S<br/>Form B: Laenge x Belaege, ggf. C -> B"]
    G --> J["Umrechnung nach p.u.<br/>r_pu, x_pu, b_shunt_pu"]
    I --> J
    E --> K["Slack-Setpoint in Rechteckform<br/>v_mag, v_ang -> slack_vr_pu, slack_vi_pu"]
    E --> L["Bus-Einspeisungen in p.u.<br/>p_spec_pu, q_spec_pu"]

    H --> M["NetworkSpec<br/>BusSpec + LineSpec + p/q + Slack-Setpoint"]
    J --> M
    K --> M
    L --> M

    M --> N["compile.compile_network(spec)"]
    N --> O["CompiledTopology<br/>- n_bus<br/>- slack_bus<br/>- from_bus<br/>- to_bus<br/>- variable_buses"]
    N --> P["NetworkParams<br/>- p_spec_pu / q_spec_pu<br/>- g_series_pu / b_series_pu<br/>- b_shunt_pu<br/>- slack_vr_pu / slack_vi_pu"]
    F --> Q["Flat-Start PFState<br/>vr_pu = 1, vi_pu = 0 fuer Nicht-Slack-Busse"]

    O --> R["Interne JAX-Netzdarstellung"]
    P --> R
    Q --> R
```

### Kurzinterpretation

- `reader.py` ist die reine Python-Eingabeschicht. Dort wird nichts in JAX
  gebaut.
- `parser.py` ist die Bruecke zwischen Rohdaten und JAX-Welt.
- Die entscheidende Trennung ist:
  - `CompiledTopology` = statische Struktur
  - `NetworkParams` = differenzierbare physikalische Parameter
  - `PFState` = differenzierbarer Solverzustand
- `compile_network()` ist der letzte Schritt vor dem numerischen Kern. Danach
  arbeitet der Grid-Core nur noch mit Arrays.

## 2. Interne JAX-Netzdarstellung -> Kern -> Solver -> implizite Differenzierung

```mermaid
flowchart TD
    A["Interne JAX-Netzdarstellung<br/>CompiledTopology + NetworkParams + PFState"] --> B["core.ybus.build_ybus()"]
    A --> C["core.residuals.state_to_voltage()"]
    B --> D["Y_bus"]
    C --> E["V = Vr + j Vi"]
    D --> F["core.residuals.calc_power_injection()<br/>I = Y_bus @ V<br/>S_calc = V * conj(I)"]
    E --> F
    F --> G["core.residuals.power_flow_residual()<br/>r = [P_spec - Re(S_calc) | Q_spec - Im(S_calc)]<br/>nur fuer Nicht-Slack-Busse"]
    G --> H["core.residuals.residual_loss()<br/>0.5 * ||r||^2"]

    G --> I["solver.newton.solve_power_flow()<br/>while_loop + jacfwd + linear solve"]
    A --> I
    I --> J["Geloester PFState"]

    J --> K["core.observables.power_flow_observables()"]
    A --> K
    K --> L["Solver-unabhaengige Outputs<br/>- |V| und Winkel<br/>- Slack P/Q<br/>- Gesamtverluste P/Q<br/>- Leitungsfluesse"]

    G --> M["solver.implicit.solve_power_flow_implicit()<br/>custom_root"]
    A --> M
    M --> N["Vorwaertsloesung:<br/>nutzt bestehenden Newton-Solver"]
    M --> O["Rueckwaertsmodus:<br/>linearisierter Root-Solve am Konvergenzpunkt<br/>kein Unrolling der Newton-Iterationen"]
    M --> P["Implizit differenzierbarer PFState"]

    P --> K
    P --> Q["validation.gradient_check.output_value()/output_vector()"]
    Q --> R["AD-Gradienten via jax.grad / jax.jacfwd"]
    A --> S["validation.finite_diff.central_difference()"]
    Q --> S
    S --> T["FD-Gradienten"]
    R --> U["Experiment 2<br/>AD vs. FD"]
    T --> U

    J --> V["validation.pandapower_ref.solve_with_jax()"]
    V --> W["Experiment 1<br/>JAX vs. pandapower"]
```

### Kurzinterpretation

- Der numerische Kern lebt in `core/`:
  - `build_ybus()`
  - `state_to_voltage()`
  - `calc_power_injection()`
  - `power_flow_residual()`
  - `residual_loss()`
- `solver/newton.py` loest das stationaere Root-Problem vorwaerts.
- `solver/implicit.py` verwendet dieselbe Residuenformulierung, aber ersetzt
  die Gradientenableitung durch `jax.lax.custom_root`.
- `core/observables.py` trennt die fachliche Auswertung bewusst vom Solver:
  dieselben Outputs koennen aus Newton- oder implicit-Loesungen berechnet werden.
- Experiment 1 nutzt die Vorwaertsloesung und vergleicht sie mit `pandapower`.
- Experiment 2 nutzt die implizite Loesung plus Observables und vergleicht
  lokale Sensitivitaeten gegen zentrale Finite Differences.

## Einordnung der beiden Experimente

### Experiment 1

Pfad:

`JSON -> RawNetwork -> parse() -> CompiledTopology / NetworkParams / PFState -> solve_power_flow() -> observables / line flows -> Vergleich mit pandapower`

Ziel:

- Validierung des stationaeren Vorwaertssolvers
- Vergleich von Spannungen, Winkeln, Verlusten und Leitungsfluessen

### Experiment 2

Pfad:

`JSON -> RawNetwork -> Szenarioanpassung (low_pv, medium_pv, high_pv) -> parse() -> solve_power_flow_implicit() -> power_flow_observables() -> jax.grad / jacfwd`

Parallel dazu:

`dieselben Outputs -> zentrale Finite Differences`

Ziel:

- Validierung lokaler Sensitivitaeten des differentiierbaren Solvers
- Vergleich von AD-Gradienten und FD-Gradienten fuer ausgewaehlte Inputs und Outputs

## Wichtigste Architekturentscheidung im Gesamtfluss

Die zentrale Softwareidee ist die strikte Trennung von drei Ebenen:

1. Menschenfreundliche Eingabeebene
   `RawNetwork`, JSON, physikalische Einheiten, Leitungsformen A/B

2. Interne JAX-Netzdarstellung
   `CompiledTopology`, `NetworkParams`, `PFState`

3. Numerisch-differenzierbare Kernlogik
   Residuen, Newton-Solver, implizite Differenzierung, Observables

Dadurch koennen spaetere Upstream-Modelle wie PV-Physik oder ein neuronales
PV-Modell denselben Grid-Core verwenden, ohne dass der numerische Kern selbst
umgebaut werden muss.
