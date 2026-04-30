# Validierungsstatus

## Überblick

Der aktuelle Stand validiert sowohl den ursprünglichen Minimalfall als auch den erweiterten `pandapower`-Hauptdemonstrator `example_simple()`.

Die wichtigsten validierten Bausteine sind:

- numerischer V1-Kern,
- JSON-/Parser-Schicht,
- `pandapower`-I/O-Pipeline,
- Leitungs-, Trafo- und Shunt-Stamps,
- Bus-Fusion und Switch-Vorverarbeitung,
- Vorwärtsvalidierung am `example_simple()`-Netz,
- Gradientenvalidierung am `example_simple()`-Netz,
- PV-Kopplungspunkt und Baseline-Kopplungsinterface.

## Validiert: Numerischer V1-Kern

Validiert wurden:

- Compiler,
- Y-Bus-Aufbau,
- Residuen,
- Newton-Solver,
- Observables,
- Finite-Difference-Helfer,
- implizite Differentiation.

Der ursprüngliche 3-Bus-PoC stimmt im Vorwärtssolve innerhalb numerischen Rundungsrauschens mit `pandapower` überein und bleibt als Minimal-Regressionstest erhalten.

## Validiert: `pandapower`-I/O-Pipeline

Umgesetzt und getestet sind:

- `from_pandapower(net) -> NetworkSpec`,
- `load_pandapower_json(path) -> NetworkSpec`,
- Bus-Fusion über geschlossene Bus-Bus-Switches,
- Deaktivierung offener Leitungen,
- Import von Bus, Slack, Load, SGen, Gen, Line, 2W-Trafo und Shunt,
- erweiterte JSON-Formate für Trafo und Shunt.

Bekannte Grenze: Der Import ist ein kontrollierter Teilumfang und keine vollständige `pandapower`-Reimplementierung.

## Validiert: Experiment 1b `example_simple()`

Ordner:

```text
experiments/results/exp01_example_simple_validation/
```

Artefakte:

- `validation_summary.csv/json`,
- `bus_results.csv/json`,
- `slack_results.csv/json`,
- `line_flows.csv/json`,
- `trafo_flows.csv/json`,
- `losses.csv/json`,
- `structure_summary.csv/json`,
- `metadata.json`,
- `README.md`.

### Umfang

Szenarien:

- `base`,
- `load_low`,
- `load_high`,
- `sgen_low`,
- `sgen_high`,
- `combined_high_load_low_sgen`,
- `combined_low_load_high_sgen`.

Referenzmodi:

- `scope_matched`: strikte Validierung, `gen -> sgen(P, Q=0)`,
- `original_pandapower`: Kontextvergleich mit originalem `pandapower`-PV-Bus.

### Ergebnisbewertung

Alle Szenarien konvergieren. `diffpf` verwendet eine trafo-shift-aware Initialisierung und erreicht Residualnormen im Bereich von ca. `4e-11`.

Im `scope_matched`-Modus sind die Knotenspannungen sehr gut validiert:

```text
max |ΔV| ≈ 6e-5 pu
max |Δθ| ≈ 0.0023°
```

Leitungsverluste stimmen sehr gut überein. Ein systematischer Offset bleibt bei Transformatorverlusten, Slack-Leistung und Trafoflüssen. Dieser liegt im Base-Fall ungefähr bei:

```text
ΔP_loss_total ≈ 0.014 MW
ΔQ_loss_total ≈ 0.029 MVAr
```

Die naheliegende Ursache ist eine nicht vollständig `pandapower`-identische Transformatorabbildung, insbesondere im Zusammenhang mit der großen 150°-Phasenverschiebung. Die Knotenzustände bleiben dennoch sehr nah an der Referenz.

Im `original_pandapower`-Modus sind größere Abweichungen erwartbar, weil `pandapower` dort den `gen` als echten spannungsregelnden PV-Bus löst, `diffpf` im aktuellen Scope jedoch nicht.

## Validiert: Experiment 2b `example_simple()`

Ordner:

```text
experiments/results/exp02_example_simple_gradients/
```

Artefakte:

- `gradient_table.csv/json`,
- `error_summary.csv/json`,
- `fd_step_study.csv/json`,
- `metadata.json`,
- `README.md`.

### Umfang

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

Gesamtumfang:

```text
3 × 4 × 4 = 48 Gradienten
```

### Ergebnisbewertung

Alle 48 Gradienten sind gültig:

- `n_valid = 48`,
- keine AD-Ausfälle,
- keine FD-Ausfälle,
- Residualnormen ca. `4e-11`,
- stabiler Newton-Solve.

Die absoluten Fehler liegen im Bereich von ca. `1e-9`. Der maximale absolute Fehler liegt bei etwa `4.7e-9`.

Relative Fehler sind überwiegend sehr klein. Höhere relative Fehler treten vor allem bei Shunt-Sensitivitäten auf, insbesondere bei sehr kleinen Gradientenbeträgen. Diese sind wegen der gleichzeitig sehr kleinen absoluten Fehler numerisch unkritisch.

Die FD-Schrittweitenstudie bestätigt das erwartete Verhalten: zu kleine Schrittweiten verstärken Rundungsfehler, während mittlere Schrittweiten stabile Vergleiche liefern.

## Validiert: PV-Kopplungspunkt und Baseline-Check

Umgesetzt:

- `src/diffpf/models/pv.py`,
- `experiments/check_pv_coupling_baseline.py`,
- `tests/test_pv_model.py`.

Festgelegt:

```text
Kopplungspunkt:     "MV Bus 2"
Ersetztes Element:  sgen "static generator"
P_ref:              2.0 MW
Q_ref:             -0.5 MVAr
Q/P:               -0.25
```

Der Baseline-Check entfernt das statische `sgen`, injiziert bei `G = 1000 W/m²` und `T_cell = 25 °C` dieselben P/Q-Werte über das JAX-kompatible Interface und vergleicht relevante Netzgrößen.

## Noch nicht vollständig validiert

- End-to-End-Experiment 3 mit variierenden Wettergrößen auf `example_simple()`.
- Modularitätsvergleich mehrerer Upstream-Modelle auf `example_simple()`.
- Gradientenbasierte gekoppelte Optimierung auf `example_simple()`.
- vollständiges PV-Bus-Enforcement für `gen`.
- Q-Limits und PV↔PQ-Umschaltung.
- vollständige `pandapower`-Trafo-Parität.
- größere oder weitere `pandapower`-Netze außerhalb des aktuellen Scopes.

## Gesamtbewertung

Das Fundament ist ausreichend zuverlässig, um mit Experiment 3 und der Integration vorgelagerter Modelle auf `example_simple()` fortzufahren.

Die wichtigsten Einschränkungen sind dokumentiert und betreffen vor allem vollständige `pandapower`-Semantik, Generator-Spannungsregelung und feine Trafo-Modellparität. Sie blockieren die geplante PV-Upstream-Kopplung nicht, da diese bewusst als P/Q-Einspeisemodell am PQ-Bus umgesetzt wird.
