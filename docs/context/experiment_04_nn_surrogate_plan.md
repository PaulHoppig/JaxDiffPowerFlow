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

Das aktuelle Hauptmodell ist ein P-only-MLP mit drei Eingaben, zwei
versteckten Schichten der Breite 16, `tanh`-Aktivierungen und einer skalaren
Ausgabe. Die Default-Konfiguration des Hauptlaufs verwendet 32768
Trainingspunkte, 8192 Validierungspunkte und einen separaten
8192-Punkte-Evaluationssplit, einen festen Seed und einfaches
Full-Batch-Gradient-Descent in JAX.

Der dokumentierte Referenz-Hauptlauf nutzt eine nicht-zyklische
Cosine-Decay-Lernrate mit Startwert `8e-2`, Endwert `1e-4` und 8000
Trainingsschritten. Im aktualisierten Hauptlauf vom 2026-05-22 lag der beste
Validation-Checkpoint am letzten Schritt `8000`; das Training war mit der
hoeheren Start-Lernrate stabil.

Der aktuelle Exp.-4-Tuninglauf nutzt einen zweiphasigen
Warm-Restart-Finetune. Da kein persistierter Parametercheckpoint des
8000-Schritt-Referenzlaufs existierte, fuehrt das Skript Phase A erneut im
Prozess aus und startet Phase B aus dem besten Phase-A-Validation-Checkpoint.
Phase A bleibt der 8000-Schritt-Cosine-Decay-Lauf `8e-2 -> 1e-4`. Phase B
nutzt 8000 zusaetzliche Full-Batch-Schritte mit
`cosine_warm_restarts_decay`:

- Zyklus 1: `2e-2 -> 5e-4` ueber 2000 Schritte,
- Zyklus 2: `1e-2 -> 2e-4` ueber 2000 Schritte,
- Zyklus 3: `5e-3 -> 1e-4` ueber 2000 Schritte,
- Zyklus 4: `2e-3 -> 5e-5` ueber 2000 Schritte.

Das globale Best-Validation-Checkpointing laeuft ueber beide Phasen. Falls der
Finetune schlechter waere, bliebe der beste Phase-A-Checkpoint final. Im
Warm-Restart-Lauf vom 2026-05-22 gewinnt Phase B im vierten Zyklus bei global
Step `16000`.

Der aktuelle Modellkapazitaetslauf vom 2026-05-22 erhoeht ausschliesslich
`hidden_width` von 8 auf 16. `hidden_layers = 2`, `tanh`, Wetterbereiche,
Input-Normalisierung, Datensatzgroessen, Loss-Funktion, P-only-Ziel und
`Q = -0.25 * P` bleiben unveraendert. Der Width-16-Lauf wird neu initialisiert
und vollstaendig trainiert; ein Width-8-Checkpoint wird wegen inkompatibler
Parameterformen nicht wiederverwendet. Die berechnete Parameterzahl steigt von
113 auf 353. Der beste globale Validation-Checkpoint liegt erneut in Phase B,
Zyklus 4, global Step `16000`.

Kleine Smoke-Test-Konfigurationen in den Tests duerfen weiterhin deutlich
weniger Punkte verwenden. Diese Mini-Konfigurationen pruefen nur Import,
Trainingsmechanik und Exportschema; sie ersetzen nicht den dokumentierten
Exp.-4-Hauptlauf.

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

Der grosse `eval`-Split dient der statistischen Surrogatfehler-Auswertung. Die
kleine feste Liste repraesentativer Wetterfaelle bleibt separat erhalten und
wird fuer Power-Flow-Modellvergleich, AD-vs-FD-Spotchecks und
Sensitivitaetsmuster genutzt. Dadurch wird nicht versehentlich ein
8192-Punkte-Power-Flow-Vergleich ueber alle Modelle und Netzszenarien gestartet.

## Bewertungsartefakte

Die Ergebnisse werden nach
`experiments/results/exp04_modular_upstream_nn_surrogate/` geschrieben.
Wesentliche Artefakte sind:

- `training_dataset_summary.csv/json`,
- `training_history.csv/json`,
- `surrogate_error_table.csv/json`,
- `model_comparison.csv/json`,
- `pf_observable_error_table.csv/json`,
- `pf_observable_error_summary.csv/json`,
- `coupling_summary.csv/json`,
- `gradient_success_table.csv/json`,
- `sensitivity_pattern_summary.csv/json`,
- `sensitivity_error_table.csv/json`,
- `sensitivity_error_summary.csv/json`,
- `training_improvement_summary.csv/json`,
- `architecture_comparison_summary.csv/json`,
- `run_summary.csv/json`,
- `metadata.json`,
- `README.md`.

`coupling_summary` dokumentiert direkt, dass alle Modelle denselben
Kopplungsbus, denselben P/Q-Adapter, denselben Power-Flow-Kern und keine
Controller-, Q-Limit- oder PV-PQ-Umschaltlogik verwenden.

`gradient_success_table` enthaelt repraesentative AD-vs-FD-Spotchecks fuer die
Wetterinputs. `sensitivity_pattern_summary` vergleicht lokale
End-to-End-Sensitivitaetsmuster zwischen analytischem Modell, NN-Surrogat und
direkter Baseline. Cosine Similarity bleibt dort als Zusatzmetrik fuer
Richtungsaehnlichkeit erhalten, ist aber nicht mehr die alleinige
Hauptaussage.

Die zentrale netzseitige Modellvergleichsaussage wird durch
`pf_observable_error_table` und `pf_observable_error_summary` gestuetzt. Diese
Tabellen vergleichen `nn_p_only_fixed_kappa` und `direct_pq_scale_baseline`
gegen `analytic_pv_weather` mit signiertem Fehler, absolutem Fehler, RMSE und
robustem relativen Fehler mit Nenner-Floor.

Die zentrale Sensitivitaetsaussage wird durch `sensitivity_error_table` und
`sensitivity_error_summary` gestuetzt. Diese Tabellen berichten absolute
Gradientenfehler, relative Gradientenfehler mit Nenner-Floor,
Vorzeichenuebereinstimmung und Magnitudenverhaeltnisse gegen
`analytic_pv_weather`.

`training_improvement_summary` vergleicht den aktuellen Width-16-Lauf mit
Phase-A/Warm-Restart-Finetune gegen den 8000-Schritt-Referenzlauf mit Val-MSE
`2.565834826e-04`, Val-MAE `0.024622580 MW`, Eval P-MAE `0.024815085 MW`,
Eval P-RMSE `0.032617826 MW` und maximalem P-Fehler `0.185804774 MW`. Der
Width-16-Lauf erreicht Val-MSE `1.0796882754565427e-04`, Val-MAE
`0.01641394628038067 MW`, Eval P-MAE `0.016544012227095853 MW`, Eval P-RMSE
`0.021139486791535485 MW` und maximalen P-Fehler
`0.11933011475462596 MW`.

`architecture_comparison_summary` vergleicht den Width-16-Lauf gegen den
bisher besten Width-8-Warm-Restart-Lauf mit Val-MSE `2.3189919622e-04`,
Val-MAE `0.023338907 MW`, Eval P-MAE `0.023523218 MW`, Eval P-RMSE
`0.031040266 MW` und maximalem P-Fehler `0.177669209 MW`. Relative
Verbesserungen des Width-16-Kandidaten sind Val-MSE `0.5344148263`, Val-MAE
`0.2967131545`, Eval P-MAE `0.2966943457`, Eval P-RMSE `0.3189656689` und
maximaler P-Fehler `0.3283579331`.

## Bewusste Vereinfachungen

Das NN ist kein grosses Prognosemodell und wird nicht auf Messdaten trainiert.
Es ist ein kleiner Distillation-Surrogatnachweis fuer die Kopplungsmodularitaet.
Das Modell gibt nur Wirkleistung frei aus; Blindleistung folgt deterministisch
aus `kappa = -0.25`. Es gibt keine Spannungsregelung, keine Q-Limits, keine
PV-PQ-Umschaltung und keine Controllerlogik. Der aktuelle
Modellkapazitaetslauf aendert nur die MLP-Breite; Datensatzgroessen,
Wetterbereiche, Input-Normalisierung, Loss-Funktion, analytisches PV-Modell,
P/Q-Kopplung und Power-Flow-Kern bleiben unveraendert. Dies ist keine neue
wissenschaftliche Fragestellung, sondern eine kontrollierte Verbesserung des
NN-Surrogats innerhalb derselben Modellgrenzen.

Die Aussage ist lokal und auf den dokumentierten Trainings- und
Auswertungsbereich beschraenkt. Die Ergebnisqualitaet des NN darf daher nicht
als allgemeine PV-Prognoseguete interpretiert werden.
