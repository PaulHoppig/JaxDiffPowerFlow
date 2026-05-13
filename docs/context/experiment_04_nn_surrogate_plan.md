# Experiment 4 - Modulare Upstream-Kopplung mit NN-PQ-Surrogat

## Ziel und Forschungsfrage

Experiment 4 demonstriert die Modularitaet des differenzierbaren
Power-Flow-Kerns. Die zentrale Frage lautet, ob derselbe stationaere
AC-Power-Flow-Kern mit unterschiedlichen vorgelagerten Modellen gekoppelt
werden kann, ohne die numerische Kernformulierung anzupassen.

Die Hypothese ist, dass analytische und neuronale Upstream-Modelle ueber eine
einheitliche P/Q-Schnittstelle an `NetworkParams.p_spec_pu` und
`NetworkParams.q_spec_pu` gekoppelt werden koennen, waehrend Konvergenz und
Differenzierbarkeit erhalten bleiben.

## Netz und Kopplungspunkt

Das Experiment verwendet `pandapower.networks.example_simple()` im
scope-matched Modus. Der aktive `gen` wird wie in Experiment 1 bis 3 zu einem
`sgen(P, Q=0)` vereinfacht. Das zu ersetzende Element ist weiterhin das
`sgen` mit Name `"static generator"` am Bus `"MV Bus 2"`.

Die Topologie bleibt statisch. Das ersetzte `sgen` wird bei der Netzbasis
deaktiviert; alle Upstream-Modelle liefern P/Q-Injektionen in MW/MVAr, die ueber
denselben Adapter `inject_pv_at_bus(...)` in das `NetworkParams`-Pytree
geschrieben werden.

## Vergleichene Upstream-Modelle

Das Referenzmodell `analytic_pv_weather` ist das bestehende analytische
PV-Wettermodell aus Experiment 3. Es verwendet NOCT-SAM fuer die Zelltemperatur
und die analytische PV-Leistungsformel.

Das Modell `nn_p_only_fixed_kappa` ist ein kleines JAX-MLP ohne Equinox,
Flax, Optax oder andere ML-Dependencies. Es nutzt die Wetterinputs
`g_poa_wm2`, `t_amb_c` und `wind_ms`, gibt eine normierte Wirkleistung aus und
setzt `Q = -0.25 * P`. Damit bleibt das Q/P-Verhaeltnis des ersetzten
`sgen "static generator"` erhalten.

Das Modell `direct_pq_scale_baseline` ist eine einfache differenzierbare
Baseline mit `P = 2.0 * g_poa_wm2 / 1000.0` und `Q = -0.25 * P`. Sie dient nur
als weitere Schnittstellenkontrolle, nicht als fachlich genaues PV-Modell.

## NN-Architektur und Training

Das Hauptmodell ist ein P-only-MLP mit drei Eingaben, zwei versteckten Schichten
der Breite 8, `tanh`-Aktivierungen und einer skalaren Ausgabe. Die Default-
Konfiguration verwendet 512 Trainings- und 128 Validierungspunkte, einen festen
Seed und einfaches Full-Batch-Gradient-Descent in JAX.

Der synthetische Trainingsdatensatz wird durch Distillation des analytischen
PV-Wettermodells erzeugt. Die Trainingsbereiche sind:

- `g_poa_wm2`: 0 bis 1200 W/m^2,
- `t_amb_c`: -10 bis 45 degC,
- `wind_ms`: 0.5 bis 10 m/s.

Die Input-Normalisierung ist:

```text
g_norm = (g_poa_wm2 - 600.0) / 600.0
t_norm = (t_amb_c - 17.5) / 27.5
w_norm = (wind_ms - 5.25) / 4.75
```

Die Zielgroesse fuer das Training ist `P_ref_mw / 2.0`.

## Bewertungsartefakte

Die Ergebnisse werden nach
`experiments/results/exp04_modular_upstream_nn_surrogate/` geschrieben.
Wesentliche Artefakte sind:

- `training_dataset_summary.csv/json`,
- `training_history.csv/json`,
- `surrogate_error_table.csv/json`,
- `model_comparison.csv/json`,
- `coupling_summary.csv/json`,
- `gradient_success_table.csv/json`,
- `sensitivity_pattern_summary.csv/json`,
- `run_summary.csv/json`,
- `metadata.json`,
- `README.md`.

`coupling_summary` dokumentiert direkt, dass alle Modelle denselben
Kopplungsbus, denselben P/Q-Adapter, denselben Power-Flow-Kern und keine
Controller-, Q-Limit- oder PV-PQ-Umschaltlogik verwenden.

`gradient_success_table` enthaelt repraesentative AD-vs-FD-Spotchecks fuer die
Wetterinputs. `sensitivity_pattern_summary` vergleicht lokale
End-to-End-Sensitivitaetsmuster zwischen analytischem Modell, NN-Surrogat und
direkter Baseline.

## Bewusste Vereinfachungen

Das NN ist kein grosses Prognosemodell und wird nicht auf Messdaten trainiert.
Es ist ein kleiner Distillation-Surrogatnachweis fuer die Kopplungsmodularitaet.
Das Modell gibt nur Wirkleistung frei aus; Blindleistung folgt deterministisch
aus `kappa = -0.25`. Es gibt keine Spannungsregelung, keine Q-Limits, keine
PV-PQ-Umschaltung und keine Controllerlogik.

Die Aussage ist lokal und auf den dokumentierten Trainings- und
Auswertungsbereich beschraenkt. Die Ergebnisqualitaet des NN darf daher nicht
als allgemeine PV-Prognoseguete interpretiert werden.
