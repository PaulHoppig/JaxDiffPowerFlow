# Experiment- und Validierungsplan

## Ziel

Die Experimente sollen zeigen, dass `diffpf`:

1. stationäre AC-Power-Flow-Zustände korrekt und robust berechnet,
2. lokal korrekt differenzierbar ist,
3. über eine `pandapower`-nahe I/O-Pipeline anschlussfähig ist,
4. vorgelagerte Modelle JAX-kompatibel koppeln kann,
5. modellübergreifende Sensitivitäten und einfache Optimierungen ermöglicht.

## Demonstratorregel

Der ursprüngliche 3-Bus-Fall bleibt als historischer Minimal- und Kontrollfall für bereits umgesetzte Basisvalidierungen erhalten.

Für neue Arbeiten gilt:

```text
Ab Experiment 3 wird ausschließlich das pandapower-Netz example_simple() verwendet.
```

Das Netz darf je Experiment angepasst werden, etwa durch Entfernen des `sgen "static generator"` und Ersetzen durch ein vorgelagertes PV-Modell. Der JAX-Kern bleibt unverändert.

## Gemeinsame Zielgrößen

Wiederkehrende netzseitige Observables:

- Spannung am Kopplungsbus `"MV Bus 2"`,
- Spannungswinkel,
- Slack-Wirk- und Blindleistung,
- Gesamtwirk- und Blindverluste,
- Leitungsflüsse, insbesondere im MV-Zweig,
- Transformator-HV-/LV-Flüsse,
- Residualnorm,
- Konvergenzstatus und Iterationszahl.

## Experiment 1: Elektrische Solver-Validierung

### Ziel

Vorwärtsvalidierung des AC-Power-Flow-Kerns gegen `pandapower`.

### Aktueller Status

Umgesetzt für:

- ursprüngliches 3-Bus-PoC,
- `pandapower example_simple()` als erweiterten Hauptdemonstrator.

### Methodik für `example_simple()`

Es werden zwei Referenzmodi verwendet:

1. `scope_matched`: `gen` wird in `sgen(P=gen.p_mw, Q=0)` umgewandelt. Dieser Modus ist die strikte numerische Validierung.
2. `original_pandapower`: `pandapower` löst das Originalnetz mit PV-Bus-Generator. Dieser Modus ist nur ein Kontextvergleich.

Szenarien:

- `base`,
- `load_low`,
- `load_high`,
- `sgen_low`,
- `sgen_high`,
- `combined_high_load_low_sgen`,
- `combined_low_load_high_sgen`.

### Zielgrößen

- Konvergenz,
- Iterationen,
- Residualnorm,
- Knotenspannungen,
- Winkel,
- Slack P/Q,
- Leitungsflüsse,
- Transformatorflüsse,
- Wirk- und Blindverluste,
- Strukturzusammenfassung.

### Artefakte

Ordner:

```text
experiments/results/exp01_example_simple_validation/
```

Dateien:

- `validation_summary.csv/json`,
- `bus_results.csv/json`,
- `slack_results.csv/json`,
- `line_flows.csv/json`,
- `trafo_flows.csv/json`,
- `losses.csv/json`,
- `structure_summary.csv/json`,
- `metadata.json`,
- `README.md`.

## Experiment 2: Gradientenvalidierung

### Ziel

Numerischer Nachweis, dass implizite AD-Gradienten mit zentralen Finite Differences übereinstimmen.

### Aktueller Status

Umgesetzt für:

- ursprüngliches 3-Bus-PoC,
- `example_simple()` im scope-matched Modell.

### Methodik für `example_simple()`

Szenarien:

- `base`,
- `load_high`,
- `sgen_high`.

Eingangsparameter:

- `load_scale_mv_bus_2`,
- `sgen_scale_static_generator`,
- `shunt_q_scale`,
- `trafo_x_scale`.

Ausgangsobservables:

- `vm_mv_bus_2_pu`,
- `p_slack_mw`,
- `total_p_loss_mw`,
- `p_trafo_hv_mw`.

Damit entstehen:

```text
3 Szenarien × 4 Eingangsparameter × 4 Ausgangsgrößen = 48 Gradienten
```

Zusätzlich wird eine kleine FD-Schrittweitenstudie für drei repräsentative Gradienten durchgeführt.

### Artefakte

Ordner:

```text
experiments/results/exp02_example_simple_gradients/
```

Dateien:

- `gradient_table.csv/json`,
- `error_summary.csv/json`,
- `fd_step_study.csv/json`,
- `metadata.json`,
- `README.md`.

## Experiment 3: Cross-Domain-Sensitivität mit PV-Upstream-Modell

**Status: umgesetzt** – siehe `experiments/exp03_cross_domain_pv_weather.py` und
`docs/context/experiment_03_plan.md`.

### Ziel

Nachweis des zentralen Mehrwerts: Sensitivitäten von vorgelagerten, nicht-elektrischen
Wettergrößen auf elektrische Zielgrößen durch die vollständige Kette Wetter → PV → Netz.

### Demonstrator

Ausschließlich `example_simple()` (scope-matched: `gen` → `sgen(P, Q=0)`).

### Kopplung

Das `sgen "static generator"` am Bus `"MV Bus 2"` wird durch das V2-PV-Wettermodell
ersetzt. Die Kette lautet:

```text
g_poa_wm2, t_amb_c, wind_ms
    -> cell_temperature_noct_sam(...)    [NOCT-SAM Zelltemperatur]
    -> pv_pq_injection_from_weather(...) [PV P/Q-Einspeisung]
    -> P_pv, Q_pv am Bus "MV Bus 2"
    -> NetworkParams
    -> AC-Power-Flow (impliziter Newton-Solver)
    -> elektrische Observables
```

### Eingangsgrößen (Pflicht)

- `g_poa_wm2` – Einstrahlungsintensität in W/m²,
- `t_amb_c` – Umgebungstemperatur in °C,
- `wind_ms` – Windgeschwindigkeit in m/s.

### Feste Modellkonstanten

`alpha = 1.0` und `kappa = -0.25` werden in Experiment 3 **nicht variiert**.
Sie sind feste Konstanten; Sensitivitäten beziehen sich ausschließlich auf Wettergrößen.

### Elektrische Betriebspunkte

- `base` (Lastfaktor 1.00),
- `load_low` (Lastfaktor 0.75),
- `load_high` (Lastfaktor 1.25).

### Wetterdesign

- **2D-Gitter:** `g_poa_wm2` ∈ {200, 400, 600, 800, 1000} W/m² × `t_amb_c` ∈
  {5, 15, 25, 35, 45} °C bei festem `wind_ms = 2.0 m/s` → 25 Fälle.
- **1D-Temperatursweep (Pflicht-Sweep):** `t_amb_c` ∈ {5, 15, 25, 35, 45, 55} °C
  bei festem `g_poa_wm2 = 800 W/m²` und `wind_ms = 2.0 m/s` → 6 Fälle.
- Gesamt: 3 Betriebspunkte × 31 Wetterfälle = **93 Forward-Solves**.

### Pflicht-Observables

- `vm_mv_bus_2_pu` (Spannungsbetrag MV Bus 2),
- `p_slack_mw` (Slack-Wirkleistung),
- `total_p_loss_mw` (Gesamtwirkleistungsverluste),
- `p_trafo_hv_mw` (HV-Trafofluss).

### Pflicht-Sensitivitäten

Lokale Gradienten per reverse-mode AD (implizite Differentiation):
`d_observable / d_g_poa_wm2`, `d_observable / d_t_amb_c`, `d_observable / d_wind_ms`
für jedes Pflicht-Observable und jeden (Betriebs-, Wetter-)punkt.

### Methodische Begrenzung

Statt einer vollständigen AD-vs-FD-Validierung (wie in Exp. 2) nur ein
**gezielter Spot-Check** für 4 repräsentative Wettergradienten (AD vs. zentrales FD).

### Artefakte

Ordner: `experiments/results/exp03_cross_domain_pv_weather/`

- `scenario_grid.csv/json`: Forward-Solve-Ergebnisse (tidy),
- `sensitivity_table.csv/json`: AD-Sensitivitäten (tidy),
- `gradient_spotcheck.csv/json`: AD-vs-FD-Spot-Check (tidy),
- `run_summary.csv/json`: Zusammenfassung pro Betriebspunkt,
- `metadata.json`,
- `README.md`.

### Visualisierungen

Nur als geplante spätere Auswertung, **nicht als Pflicht-Artefakt dieses Schritts**:

- Kurven `t_amb_c -> vm_mv_bus_2_pu` für verschiedene Betriebspunkte,
- Heatmap `g_poa_wm2 × t_amb_c -> Sensitivität`,
- Sensitivitätsbalkendiagramm pro Wettereingangsgröße.

Detailplan: `docs/context/experiment_03_plan.md`.

## Experiment 4: Modularität der Modellkopplung

### Ziel

Nachweis, dass verschiedene vorgelagerte Modelle über dieselbe Kopplungsstruktur an den unveränderten Power-Flow-Kern angebunden werden können.

### Demonstrator

Ausschließlich `example_simple()`.

### Mögliche Modelle

- analytisches PV-Modell,
- thermisches Last-/Wärmepumpenmodell als zusätzliche oder skalierte Last am MV-Netz,
- einfaches neuronales Ersatzmodell,
- direkte parametrisierte P/Q-Injektion als Baseline.

### Bewertungsgrößen

- notwendige Änderungen am PF-Kern: ja/nein,
- Konvergenz,
- erfolgreiche Gradientenberechnung,
- einheitliches Kopplungsinterface,
- Vergleich der Sensitivitätsmuster.

### Artefakte

Empfohlen:

- `model_comparison.csv/json`,
- `coupling_summary.csv/json`,
- `gradient_success_table.csv/json`,
- `metadata.json`,
- `README.md`.

## Experiment 5: Gradientenbasierte gekoppelte Optimierung

### Ziel

Demonstration, dass vorgelagerte Variablen gradientenbasiert so angepasst werden können, dass eine elektrische Zielgröße erreicht wird.

### Demonstrator

Ausschließlich `example_simple()`.

### Mögliche Optimierungsvariablen

- PV-Curtailment-Faktor,
- `q_over_p` oder Leistungsfaktor der PV-Einspeisung,
- Temperatur- oder Einstrahlungsoffset,
- Parameter eines einfachen Upstream-Ersatzmodells.

### Zielfunktionen

Beispiele:

```text
J = (|V_MV Bus 2| - V_target)^2 + lambda * regularization
J = (P_slack - P_target)^2 + lambda * regularization
J = (P_trafo_hv - P_target)^2 + lambda * regularization
```

### Zielgrößen

- Optimierungsverlauf der Zielfunktion,
- Verlauf der optimierten Upstream-Variable,
- erreichte elektrische Zielgröße,
- Konvergenz des Power Flow je Optimierungsschritt,
- Plausibilität der resultierenden P/Q-Injektion.

### Artefakte

Empfohlen:

- `optimization_trace.csv/json`,
- `final_solution.json`,
- `constraint_diagnostics.csv/json`,
- `metadata.json`,
- `README.md`.

## Experiment 6: Pandapower-I/O- und Strukturvalidierung

### Ziel

Nachweis, dass die `pandapower`-I/O-Pipeline den unterstützten Modellumfang korrekt in die interne Repräsentation überführt.

### Demonstrator

Primär `example_simple()`.

### Prüfungen

- Anzahl Originalbusse vs. interne Busse nach Fusion,
- Slack-Zuordnung,
- Last-/sgen-/gen-Aggregation,
- Leitungsaktivierung nach Switch-Handling,
- Trafo- und Shunt-Mapping,
- Fehlermeldungen bei nicht unterstützten aktiven Elementen,
- Reproduzierbarkeit des Imports.

### Artefakte

Teilweise bereits in Experiment 1 enthalten:

- `structure_summary.csv/json`,
- `metadata.json`,
- zusätzliche negative Importtests bei Bedarf.

## Gesamtlogik

Experiment 1 und 2 sichern den numerischen Kern und die Gradientenbasis ab. Experiment 3 bis 5 zeigen den eigentlichen wissenschaftlichen Mehrwert auf dem `example_simple()`-Hauptdemonstrator. Experiment 6 dokumentiert die Anschlussfähigkeit an `pandapower` und schützt die I/O-Pipeline gegen Regressionen.
