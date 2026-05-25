# Changelog

## 2026-05-22 - Experiment 4 Modellkapazitaetslauf mit hidden_width = 16

### Kurzbeschreibung
- Experiment 4 wurde als kontrollierter Modellkapazitaetslauf erneut
  ausgefuehrt.
- Die NN-Architektur aendert ausschliesslich `hidden_width=8` auf
  `hidden_width=16`; `hidden_layers=2`, `tanh`, P-only und
  `Q = -0.25 * P` bleiben unveraendert.
- Datensatzgroessen bleiben `32768/8192/8192`; Wetterbereiche,
  Input-Normalisierung, Trainingsziel `P_ref_mw / 2.0`, Loss-Funktion,
  analytisches PV-Modell und Power-Flow-Kern bleiben unveraendert.
- Die Trainingsstrategie bleibt zweiphasig: Phase A mit nicht-zyklischem
  Cosine Decay `8e-2 -> 1e-4` ueber 8000 Schritte, Phase B mit 8000
  Warm-Restart-Finetune-Schritten in vier Zyklen
  (`2e-2 -> 5e-4`, `1e-2 -> 2e-4`, `5e-3 -> 1e-4`,
  `2e-3 -> 5e-5`).
- Der Width-16-Lauf wird neu initialisiert und vollstaendig trainiert; ein
  Width-8-Checkpoint wird wegen inkompatibler Parameterformen nicht
  wiederverwendet.
- Die berechnete Parameterzahl steigt von `113` auf `353`.

### Geaenderte Dateien
- `src/diffpf/models/pq_surrogate.py` - Default-MLP-Breite auf 16 gesetzt und
  Architekturkonstanten ergaenzt.
- `src/diffpf/models/__init__.py` - Architekturkonstanten exportiert.
- `experiments/exp04_modular_upstream_nn_surrogate.py` -
  Trainingshistorie um `hidden_width`/`parameter_count` erweitert,
  Width-8-Referenzkonstanten und `architecture_comparison_summary` ergaenzt,
  Metadata/README-Export auf den Width-16-Kapazitaetslauf aktualisiert.
- `experiments/plot_exp04_training_figures.py` - optionale
  Architekturvergleichsfigur `fig04_architecture_comparison` ergaenzt.
- `tests/test_pq_surrogate_model.py` - Width-16-Default und Parameterzahl
  abgesichert.
- `tests/test_exp04_modular_surrogate_outputs.py` - Exp.-4-Schema,
  Architekturvergleich und Metadata/README-Export fuer Width 16 abgesichert.
- `tests/test_exp04_plot_outputs.py` - Plot-Pipeline fuer die neue
  Architekturvergleichsfigur abgesichert.
- `docs/context/experiment_04_nn_surrogate_plan.md` - Width-16-Lauf,
  Vergleich zu Width 8 und unveraenderte Modellgrenzen dokumentiert.
- `CHANGELOG.md` - dieser Eintrag.

### Aktualisierte Artefakte
- `experiments/results/exp04_modular_upstream_nn_surrogate/metadata.json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/README.md`
- `experiments/results/exp04_modular_upstream_nn_surrogate/training_dataset_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/training_history.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/surrogate_error_table.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/model_comparison.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/coupling_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/gradient_success_table.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/sensitivity_pattern_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/pf_observable_error_table.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/pf_observable_error_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/sensitivity_error_table.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/sensitivity_error_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/training_improvement_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/architecture_comparison_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/run_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/figures/README.md`
- `experiments/results/exp04_modular_upstream_nn_surrogate/figures/fig01_training_loss_curve.png/pdf`
- `experiments/results/exp04_modular_upstream_nn_surrogate/figures/fig02_training_mae_curve.png/pdf`
- `experiments/results/exp04_modular_upstream_nn_surrogate/figures/fig03_learning_rate_schedule.png/pdf`
- `experiments/results/exp04_modular_upstream_nn_surrogate/figures/fig04_architecture_comparison.png/pdf`

### Ergebnisnotizen
- Voller Exp.-4-Lauf erfolgreich mit Python 3.11:
  `python experiments/exp04_modular_upstream_nn_surrogate.py`.
- Plot-Pipeline erfolgreich aus vorhandenen Artefakten ausgefuehrt:
  `python experiments/plot_exp04_training_figures.py`.
- Der Width-16-Lauf war stabil; es trat kein NaN/Inf-Loss auf.
- Best-Validation-Checkpoint: Phase `warm_restart_finetune`, Zyklus `4`,
  global Step `16000`, Validation-MSE `1.0796882754565427e-04`,
  Validation-MAE `0.01641394628038067 MW`.
- Finale Fehler am letzten Schritt sind identisch mit dem Best-Checkpoint:
  Train-MSE `1.0956320206449241e-04`, Val-MSE
  `1.0796882754565427e-04`, Train-MAE `0.01646623242692842 MW`,
  Val-MAE `0.01641394628038067 MW`.
- Eval-Split: P-MAE `0.016544012227095853 MW`, P-RMSE
  `0.021139486791535485 MW`, maximale P-Abweichung
  `0.11933011475462596 MW`.
- Gegenueber dem bisherigen Width-8-Warm-Restart-Lauf
  (Val-MSE `2.3189919622e-04`, Val-MAE `0.023338907 MW`, Eval P-MAE
  `0.023523218 MW`, Eval P-RMSE `0.031040266 MW`, Max P-Fehler
  `0.177669209 MW`) ist Width 16 besser.
- Relative Verbesserungen gegen Width 8: Val-MSE `0.5344148263315862`,
  Val-MAE `0.29671315454572617`, Eval P-MAE `0.2966943456845125`,
  Eval P-RMSE `0.31896566893030215`, Max P-Fehler
  `0.3283579331147584`.
- Alle drei Upstream-Modelle konvergierten in allen bestehenden kompakten
  Power-Flow-Vergleichsfaellen: `18/18` je Modell. Es wurden keine neuen
  massiven Power-Flow-Solves auf dem 8192er-Eval-Split eingefuehrt.

### Tests
- Ausgefuehrt:
  `python -m pytest tests/test_pq_surrogate_model.py -q`
  (`8 passed`).
- Ausgefuehrt:
  `python -m pytest tests/test_exp04_modular_surrogate_outputs.py -q`
  (`17 passed`).
- Ausgefuehrt:
  `python -m pytest tests/test_exp04_plot_outputs.py -q`
  (`17 passed`).
- Ausgefuehrt:
  `python -m pytest tests/test_pv_model.py -q`
  (`16 passed`).
- Optional ausgefuehrt:
  `python -m pytest tests/test_exp03_cross_domain_outputs.py -q`
  (`27 passed`).

### Bekannte Einschraenkungen
- Das NN bleibt ein synthetisches Distillation-Surrogat und kein
  Messdaten-Prognosemodell.
- Das NN bleibt P-only; `Q = -0.25 * P` bleibt deterministisch gekoppelt.
- Keine Aenderung an Datensatzgroessen, Wetterbereichen, Input-Normalisierung,
  Loss-Funktion oder Anzahl versteckter Schichten.
- Kein Gradient-Matching-Loss und keine physikalische Nebenbedingung.
- Keine Aenderung am AC-Power-Flow-Kern, Residuum, Y-Bus, Solver,
  impliziter Differentiation oder analytischen PV-Modell.
- Keine Controllerlogik, keine Q-Limits, keine PV-PQ-Umschaltung und keine
  echte PV-Bus-Spannungsregelung.
- Sensitivitaetsfehler werden weiterhin auf den kompakten Exp.-4-Faellen
  bewertet; kein 8192er-Power-Flow-Sweep wurde eingefuehrt.

## 2026-05-22 - Experiment 4 Warm-Restart-Finetune mit abnehmender zyklischer Lernrate

### Kurzbeschreibung
- Experiment 4 wurde als kontrollierter zweiphasiger Tuninglauf erneut
  ausgefuehrt.
- Da kein persistierter Parametercheckpoint des vorherigen 8000-Step-Laufs
  existierte, wurde kein echtes Laden von Platte umgesetzt. Stattdessen fuehrt
  das Experiment Phase A erneut aus und startet Phase B im selben Prozess aus
  dem besten Phase-A-Validation-Checkpoint.
- Datensatzgroessen bleiben `32768/8192/8192`; Architektur bleibt
  `hidden_width=8`, `hidden_layers=2`, `tanh`, P-only und
  `Q = -0.25 * P`.
- Phase A: nicht-zyklischer Cosine Decay `8e-2 -> 1e-4` ueber 8000 Schritte.
- Phase B: `cosine_warm_restarts_decay` mit 4 Zyklen ueber weitere 8000
  Schritte: `2e-2 -> 5e-4`, `1e-2 -> 2e-4`, `5e-3 -> 1e-4`,
  `2e-3 -> 5e-5`, jeweils ueber 2000 Schritte.
- Globales Best-Validation-Checkpointing laeuft ueber beide Phasen; alle
  nachgelagerten Surrogat-, Power-Flow- und Sensitivitaetsartefakte verwenden
  den besten globalen Validation-Checkpoint.

### Geaenderte Dateien
- `src/diffpf/models/pq_surrogate.py` - Warm-Restart-Konfigurationsfelder und
  Default-Tuningmodus fuer Experiment 4 ergaenzt.
- `src/diffpf/models/__init__.py` - Warm-Restart-Schedule-Konstante exportiert.
- `experiments/exp04_modular_upstream_nn_surrogate.py` -
  `cosine_warm_restarts_decay`, zweiphasiges Training, erweiterte
  Trainingshistorie, globale Checkpoint-Diagnostik und aktualisierte
  Improvement-Summary.
- `experiments/plot_exp04_training_figures.py` - Plot-Pipeline nutzt
  `global_step`, markiert Phase-A/Phase-B-Grenze und Warm-Restart-Zyklen.
- `tests/test_pq_surrogate_model.py` - Default-Konfiguration und
  Warm-Restart-Felder abgesichert.
- `tests/test_exp04_modular_surrogate_outputs.py` - Schedule-Grenzen,
  zweiphasige History-Spalten, Metadata/README und Improvement-Summary
  abgesichert.
- `tests/test_exp04_plot_outputs.py` - Plot-Tests fuer `global_step`, Phasen
  und Warm-Restart-History aktualisiert.
- `docs/context/experiment_04_nn_surrogate_plan.md` - Warm-Restart-Finetune,
  Referenzvergleich und unveraenderte Modellgrenzen dokumentiert.
- `CHANGELOG.md` - dieser Eintrag.

### Aktualisierte Artefakte
- `experiments/results/exp04_modular_upstream_nn_surrogate/metadata.json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/README.md`
- `experiments/results/exp04_modular_upstream_nn_surrogate/training_dataset_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/training_history.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/surrogate_error_table.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/model_comparison.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/coupling_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/gradient_success_table.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/sensitivity_pattern_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/pf_observable_error_table.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/pf_observable_error_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/sensitivity_error_table.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/sensitivity_error_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/training_improvement_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/run_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/figures/README.md`
- `experiments/results/exp04_modular_upstream_nn_surrogate/figures/fig01_training_loss_curve.png/pdf`
- `experiments/results/exp04_modular_upstream_nn_surrogate/figures/fig02_training_mae_curve.png/pdf`
- `experiments/results/exp04_modular_upstream_nn_surrogate/figures/fig03_learning_rate_schedule.png/pdf`

### Ergebnisnotizen
- Voller Exp.-4-Lauf erfolgreich mit Python 3.11:
  `python experiments/exp04_modular_upstream_nn_surrogate.py`.
- Plot-Pipeline erfolgreich aus vorhandener `training_history.csv` ausgefuehrt:
  `python experiments/plot_exp04_training_figures.py`.
- Warm-Restart-Finetune war stabil; es trat kein NaN/Inf-Loss auf.
- Best-Validation-Checkpoint: Phase `warm_restart_finetune`, Zyklus `4`,
  global Step `16000`, Validation-MSE `2.318991962219021e-04`,
  Validation-MAE `0.02333890667386509 MW`.
- Finale Fehler am letzten Schritt sind identisch mit dem Best-Checkpoint:
  Train-MSE `2.3887884675560686e-04`, Val-MSE
  `2.318991962219021e-04`, Train-MAE `0.02359521216551384 MW`,
  Val-MAE `0.02333890667386509 MW`.
- Eval-Split: P-MAE `0.023523218447294967 MW`, P-RMSE
  `0.03104026574651338 MW`, maximale P-Abweichung
  `0.1776692089622296 MW`.
- Gegenueber dem 8000-Step-Referenzlauf
  (Val-MSE `2.565834826e-04`, Val-MAE `0.024622580 MW`, Eval P-MAE
  `0.024815085 MW`, Eval P-RMSE `0.032617826 MW`, Max P-Fehler
  `0.185804774 MW`) ist der Warm-Restart-Finetune besser.
- Relative Verbesserungen: Val-MSE `0.09620372335728015`, Val-MAE
  `0.05213398945743757`, Eval P-MAE `0.05205972708556242`, Eval P-RMSE
  `0.04836497237696414`, Max P-Fehler `0.04378555438930976`.
- Alle drei Upstream-Modelle konvergierten in allen bestehenden kompakten
  Power-Flow-Vergleichsfaellen: `18/18` je Modell. Es wurden keine neuen
  massiven Power-Flow-Solves auf dem 8192er-Eval-Split eingefuehrt.

### Tests
- Ausgefuehrt:
  `python -m pytest tests/test_pq_surrogate_model.py -q`
  (`8 passed`).
- Ausgefuehrt:
  `python -m pytest tests/test_exp04_modular_surrogate_outputs.py -q`
  (`16 passed`).
- Ausgefuehrt:
  `python -m pytest tests/test_exp04_plot_outputs.py -q`
  (`15 passed`).
- Ausgefuehrt:
  `python -m pytest tests/test_pv_model.py -q`
  (`16 passed`).
- Optional ausgefuehrt:
  `python -m pytest tests/test_exp03_cross_domain_outputs.py -q`
  (`27 passed`).

### Bekannte Einschraenkungen
- Das NN bleibt ein synthetisches Distillation-Surrogat und kein
  Messdaten-Prognosemodell.
- Das NN bleibt P-only; `Q = -0.25 * P` bleibt deterministisch gekoppelt.
- Keine Aenderung an Datensatzgroessen, Wetterbereichen, Input-Normalisierung,
  NN-Architektur oder Loss-Funktion.
- Kein Gradient-Matching-Loss und keine physikalische Nebenbedingung.
- Keine Aenderung am AC-Power-Flow-Kern, Residuum, Y-Bus, Solver,
  impliziter Differentiation oder analytischen PV-Modell.
- Keine Controllerlogik, keine Q-Limits, keine PV-PQ-Umschaltung und keine
  echte PV-Bus-Spannungsregelung.
- Sensitivitaetsfehler werden weiterhin auf den kompakten Exp.-4-Faellen
  ausgewertet; der 8192er-Eval-Split wird nicht fuer massive PF-Solves genutzt.

## 2026-05-22 - Experiment 4 erneutes NN-Surrogat-Training mit staerkerem LR-Decay

### Kurzbeschreibung
- Experiment 4 wurde erneut mit unveraenderten Datensatzgroessen
  `32768/8192/8192` ausgefuehrt.
- Das JAX-only P-only-MLP nutzt weiterhin dieselbe Architektur, dieselbe
  Wetterverteilung, dieselbe Input-Normalisierung und dasselbe Trainingsziel
  `P_ref_mw / 2.0`.
- Das Optimierungsregime des Full-/Default-Hauptlaufs wurde auf
  nicht-zyklischen Cosine Decay von `8e-2` auf `1e-4` ueber `8000`
  Trainingsschritte geaendert. Best-Validation-Checkpointing bleibt aktiv.
- Es wurden keine Aenderungen am analytischen PV-Modell, am P/Q-Adapter, an
  `example_simple()`, am AC-Power-Flow-Kern, an Residuen, Y-Bus, Solver,
  impliziter Differentiation oder Observables vorgenommen.

### Geaenderte Dateien
- `src/diffpf/models/pq_surrogate.py` - Default-Lernraten und
  Default-Trainingsschritte fuer den Exp.-4-Hauptlauf auf `8e-2 -> 1e-4`
  ueber `8000` Schritte gesetzt.
- `experiments/exp04_modular_upstream_nn_surrogate.py` -
  Vorreferenzmetriken, nicht-stille Non-Finite-Trainingserkennung,
  `training_improvement_summary.csv/json`, Metadata- und README-Erweiterung.
- `tests/test_pq_surrogate_model.py` - Tests fuer die neue Full-Default-
  Konfiguration.
- `tests/test_exp04_modular_surrogate_outputs.py` - Tests fuer neue
  LR-Defaults, Learning-Rate-Export, Improvement-Summary und README/Metadata.
- `tests/test_exp04_plot_outputs.py` - Testdaten fuer die neue LR-Schedule
  aktualisiert.
- `docs/context/experiment_04_nn_surrogate_plan.md` - neue LR-Konfiguration,
  unveraenderte Modellgrenzen und Ergebnisvergleich dokumentiert.
- `CHANGELOG.md` - dieser Eintrag.

### Aktualisierte Artefakte
- `experiments/results/exp04_modular_upstream_nn_surrogate/metadata.json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/README.md`
- `experiments/results/exp04_modular_upstream_nn_surrogate/training_dataset_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/training_history.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/surrogate_error_table.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/model_comparison.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/coupling_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/gradient_success_table.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/sensitivity_pattern_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/pf_observable_error_table.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/pf_observable_error_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/sensitivity_error_table.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/sensitivity_error_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/run_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/figures/README.md`
- `experiments/results/exp04_modular_upstream_nn_surrogate/figures/fig01_training_loss_curve.png/pdf`
- `experiments/results/exp04_modular_upstream_nn_surrogate/figures/fig02_training_mae_curve.png/pdf`
- `experiments/results/exp04_modular_upstream_nn_surrogate/figures/fig03_learning_rate_schedule.png/pdf`

### Neue Artefakte
- `experiments/results/exp04_modular_upstream_nn_surrogate/training_improvement_summary.csv/json`

### Ergebnisnotizen
- Voller Exp.-4-Lauf erfolgreich mit Python 3.11:
  `python experiments/exp04_modular_upstream_nn_surrogate.py`.
- Plot-Pipeline erfolgreich aus vorhandener `training_history.csv` ausgefuehrt:
  `python experiments/plot_exp04_training_figures.py`.
- Das Training mit Start-LR `8e-2` war stabil; es trat kein NaN/Inf-Loss auf.
- Best-Validation-Checkpoint: Schritt `8000`, Validation-MSE
  `2.5658348260553096e-04`, Validation-MAE
  `0.024622580033535818 MW`.
- Finale Fehler am letzten Schritt sind identisch mit dem Best-Checkpoint:
  Train-MSE `2.6436741913525146e-04`, Val-MSE
  `2.5658348260553096e-04`, Train-MAE `0.024900537295140097 MW`,
  Val-MAE `0.024622580033535818 MW`.
- Eval-Split: P-MAE `0.02481508538556336 MW`, P-RMSE
  `0.032617825812151176 MW`, maximale P-Abweichung
  `0.18580477434619525 MW`.
- Gegenueber dem vorherigen Lauf
  (Val-MSE `8.4781e-04`, Val-MAE `0.046857 MW`, Eval P-MAE
  `0.046775 MW`, Eval P-RMSE `0.058539 MW`, Max P-Fehler
  `0.268626 MW`) ist der neue Lauf in allen genannten Surrogatmetriken besser.
- Relative Verbesserungen: Val-MSE `0.6973573293479306`, Val-MAE
  `0.4745165069565739`, Eval P-MAE `0.46947973520976244`, Eval P-RMSE
  `0.44280179346843684`, Max P-Fehler `0.3083142571970127`.
- Alle drei Upstream-Modelle konvergierten in allen bestehenden kompakten
  Power-Flow-Vergleichsfaellen: `18/18` je Modell. Es wurden keine neuen
  massiven Power-Flow-Solves auf dem 8192er-Eval-Split eingefuehrt.

### Tests
- Ausgefuehrt:
  `python -m pytest tests/test_pq_surrogate_model.py -q`
  (`8 passed`).
- Ausgefuehrt:
  `python -m pytest tests/test_exp04_modular_surrogate_outputs.py -q`
  (`14 passed`).
- Ausgefuehrt:
  `python -m pytest tests/test_exp04_plot_outputs.py -q`
  (`15 passed`).
- Ausgefuehrt:
  `python -m pytest tests/test_pv_model.py -q`
  (`16 passed`).
- Optional ausgefuehrt:
  `python -m pytest tests/test_exp03_cross_domain_outputs.py -q`
  (`27 passed`).

### Bekannte Einschraenkungen
- Das NN bleibt ein synthetisches Distillation-Surrogat und kein
  Messdaten-Prognosemodell.
- Das NN bleibt P-only; `Q = -0.25 * P` bleibt deterministisch gekoppelt.
- Keine Aenderung am AC-Power-Flow-Kern, Residuum, Y-Bus, Solver,
  impliziter Differentiation oder analytischen PV-Modell.
- Keine Controllerlogik, keine Q-Limits, keine PV-PQ-Umschaltung und keine
  echte PV-Bus-Spannungsregelung.
- Sensitivitaetsfehler werden weiterhin auf den kompakten Exp.-4-Faellen
  ausgewertet; der 8192er-Eval-Split wird nicht fuer massive PF-Solves genutzt.

## 2026-05-20 - Experiment 4 groesseres NN-Surrogat mit LR-Decay und Fehlerhauptmetriken

### Kurzbeschreibung
- Experiment 4 wurde auf eine 4-fach groessere synthetische
  Distillation-Hauptlauf-Konfiguration umgestellt:
  `32768` Trainingspunkte, `8192` Validierungspunkte und `8192`
  Evaluationspunkte.
- Das JAX-only P-only-MLP nutzt weiterhin Full-Batch-Gradient-Descent, nun mit
  nicht-zyklischem Cosine-Decay von `5e-2` auf `5e-4` ueber `4000`
  Trainingsschritte.
- Best-Validation-Checkpointing wurde ergaenzt. Die final exportierten
  Surrogatfehler und der Power-Flow-Modellvergleich verwenden die Parameter
  mit dem niedrigsten Validation-MSE.
- Der Power-Flow-Modellvergleich wurde um zentrale absolute
  Observable-Fehlermetriken gegen `analytic_pv_weather` erweitert.
- Der Sensitivitaetsvergleich wurde um absolute Gradientenfehler, robuste
  relative Gradientenfehler mit Nenner-Floor, Vorzeichenuebereinstimmung und
  Magnitudenverhaeltnisse erweitert. Cosine Similarity bleibt nur als
  Zusatzmetrik fuer Richtungsaehnlichkeit erhalten.

### Geaenderte Dateien
- `src/diffpf/models/pq_surrogate.py` - Default-Datensatzgroessen auf
  `32768/8192/8192` gesetzt und LR-Schedule-Defaults ergaenzt.
- `src/diffpf/models/__init__.py` - neue Exp.-4-Default-Konstanten exportiert.
- `experiments/exp04_modular_upstream_nn_surrogate.py` - Cosine-Decay,
  Best-Validation-Checkpointing, neue PF-Observable-Fehlerartefakte,
  neue Sensitivitaetsfehlerartefakte, erweiterte Metadata/README/Run-Summary.
- `experiments/plot_exp04_training_figures.py` - neue Fig. 3 fuer die
  exportierte Learning-Rate-Schedule.
- `tests/test_pq_surrogate_model.py` - Tests fuer neue Default-Konfiguration.
- `tests/test_exp04_modular_surrogate_outputs.py` - Tests fuer LR-Schedule,
  Smoke-Konfiguration, neue Artefaktschemas und Fehleraggregationen.
- `tests/test_exp04_plot_outputs.py` - Tests fuer die neue LR-Figur.
- `docs/context/experiment_04_nn_surrogate_plan.md` - Hauptlaufgroessen,
  LR-Decay, Best-Checkpointing und neue Hauptmetriken dokumentiert.
- `CHANGELOG.md` - dieser Eintrag.

### Aktualisierte Artefakte
- `experiments/results/exp04_modular_upstream_nn_surrogate/metadata.json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/README.md`
- `experiments/results/exp04_modular_upstream_nn_surrogate/training_dataset_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/training_history.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/surrogate_error_table.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/model_comparison.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/coupling_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/gradient_success_table.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/sensitivity_pattern_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/run_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/figures/README.md`
- `experiments/results/exp04_modular_upstream_nn_surrogate/figures/fig01_training_loss_curve.png/pdf`
- `experiments/results/exp04_modular_upstream_nn_surrogate/figures/fig02_training_mae_curve.png/pdf`

### Neue Artefakte
- `experiments/results/exp04_modular_upstream_nn_surrogate/pf_observable_error_table.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/pf_observable_error_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/sensitivity_error_table.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/sensitivity_error_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/figures/fig03_learning_rate_schedule.png/pdf`

### Ergebnisnotizen
- Voller Exp.-4-Lauf erfolgreich mit Python 3.11:
  `python experiments/exp04_modular_upstream_nn_surrogate.py`.
- Alle drei Upstream-Modelle konvergierten in allen kompakten
  Power-Flow-Vergleichsfaellen: `18/18` je Modell.
- `surrogate_error_table` enthaelt `49152` Zeilen:
  train `32768`, val `8192`, eval `8192`.
- Best-Checkpoint: Schritt `4000`, Validation-MSE `8.478118983363279e-04`,
  Validation-MAE `0.046857078471898586 MW`.
- Finale Fehler am letzten Schritt sind identisch mit dem Best-Checkpoint:
  Train-MSE `8.675546077813411e-04`, Val-MSE `8.478118983363279e-04`,
  Train-MAE `0.047145536958641356 MW`, Val-MAE
  `0.046857078471898586 MW`.
- Eval-Split: P-MAE `0.046775376273178695 MW`, P-RMSE
  `0.058538922866291296 MW`, maximale P-Abweichung
  `0.2686258793470051 MW`.
- Gegenueber dem vorherigen dokumentierten Hauptlauf
  (Val-MSE ca. `0.001756`, Val-MAE ca. `0.06663 MW`) ist der neue Lauf
  deutlich besser.
- Der grosse Eval-Split wurde nur fuer Surrogatfehler genutzt. Power-Flow- und
  Sensitivitaetsvergleiche blieben auf den bestehenden kompakten Wetterfaellen.

### Tests
- Ausgefuehrt:
  `python -m pytest tests/test_pq_surrogate_model.py -q`
  (`8 passed`).
- Ausgefuehrt:
  `python -m pytest tests/test_exp04_modular_surrogate_outputs.py -q`
  (`13 passed`).
- Ausgefuehrt:
  `python -m pytest tests/test_exp04_plot_outputs.py -q`
  (`15 passed`).
- Ausgefuehrt:
  `python -m pytest tests/test_pv_model.py -q`
  (`16 passed`).
- Optional ausgefuehrt:
  `python -m pytest tests/test_exp03_cross_domain_outputs.py -q`
  (`27 passed`).

### Bekannte Einschraenkungen
- Das NN bleibt ein synthetisches Distillation-Surrogat und kein
  Messdaten-Prognosemodell.
- Das NN bleibt P-only; `Q = -0.25 * P` bleibt deterministisch gekoppelt.
- Keine Aenderung am AC-Power-Flow-Kern, Residuum, Y-Bus, Solver,
  impliziter Differentiation oder analytischen PV-Modell.
- Keine Controllerlogik, keine Q-Limits, keine PV-PQ-Umschaltung und keine
  echte PV-Bus-Spannungsregelung.
- Sensitivitaetsfehler werden weiterhin auf den kompakten Exp.-4-Faellen
  ausgewertet; der 8192er-Eval-Split wird nicht fuer massive PF-Solves genutzt.

## 2026-05-19 - Experiment 2 erneut ausgefuehrt nach Kern-/Modellierungsaenderungen

### Kurzbeschreibung
- Experiment 2 (48 AD-vs-FD-Gradienten, `example_simple()`) wurde nach den
  Kern-/Modellierungsaenderungen (korrigierte Trafo-Magnetisierungsadmittanz,
  scope-matched Open-Line-Policy) vollstaendig neu ausgefuehrt.
- Alle 48 Gradienten konvergieren mit aktuellem Codezustand. AD und FD
  stimmen ueberein; der maximale relative Fehler ueber alle Kombinationen
  betraegt ca. 4.51e-05 (shunt_q_scale, erklaerbar durch kleinen Gradienten).
  Der Median-Relativfehler liegt deutlich darunter.
- Standard-FD-Schrittweite des Hauptlaufs auf h = 1e-3 gesetzt
  (vorher 1e-4); Fehler verbessern sich konsistent.
- Eine neue detaillierte FD-Schrittweiten-Studie (h = 1e0 ... 1e-10,
  11 Schritte, 33 Zeilen) wurde fuer drei repraesentative Gradienten ergaenzt.
- Eine FD-vs-FD-Stabilitaetsdiagnose auf Basis der detaillierten Studie wurde
  ergaenzt (30 benachbarte Schrittweitenpaare).
- Neue Figuren 8 und 9 wurden erzeugt.
- Bestehende Figuren 1-7 wurden aus neuen Artefakten aktualisiert.

### Geaenderte Dateien
- `experiments/exp02_validate_gradients_example_simple.py` - Neue Konstante
  `FD_STEPS_DETAILED`, neue Dataclasses `StepStudyDetailedRow` und
  `StepStudyDetailedSummaryRow`, neue Funktionen `build_fd_step_study_detailed()`
  und `build_fd_step_study_detailed_summary()`, aktualisiertes `export_all()`,
  `write_metadata()`, `write_readme()` und `main()`.
- `experiments/plot_exp02_gradient_figures.py` - Neue Funktionen
  `write_fd_vs_fd_step_stability_detailed()`, `plot_fd_step_study_detailed()`,
  `build_fd_vs_fd_step_stability_detailed_figure()`,
  `plot_fd_vs_fd_step_stability_detailed()`. Aktualisierte `generate_figures()`
  und `write_figures_readme()`.
- `tests/test_exp02_example_simple_gradients_outputs.py` - Neue Tests fuer
  detaillierte Artefakte und kanonische Ergebnisverzeichnis-Checks.
- `tests/test_exp02_plot_outputs.py` - Neue Tests fuer Fig. 8, Fig. 9,
  detaillierte FD-vs-FD-Stabilitaet und aktualisierter README-Test.

### Aktualisierte Artefakte (kanonisches Ergebnisverzeichnis)
- `experiments/results/exp02_example_simple_gradients/gradient_table.csv/json`
- `experiments/results/exp02_example_simple_gradients/error_summary.csv/json`
- `experiments/results/exp02_example_simple_gradients/fd_step_study.csv/json`
- `experiments/results/exp02_example_simple_gradients/metadata.json`
- `experiments/results/exp02_example_simple_gradients/README.md`

### Neue detaillierte StepStudy-Artefakte
- `experiments/results/exp02_example_simple_gradients/fd_step_study_detailed.csv/json`
- `experiments/results/exp02_example_simple_gradients/fd_step_study_detailed_summary.csv/json`
- `experiments/results/exp02_example_simple_gradients/figures/fd_vs_fd_step_stability_detailed.csv/json`

### Neue/aktualisierte Figuren
- `figures/fig01_ad_vs_fd_parity_by_observable.png/pdf` (aktualisiert)
- `figures/fig01a_ad_vs_fd_parity_global.png/pdf` (aktualisiert)
- `figures/fig02_gradient_error_heatmap.png/pdf` (aktualisiert)
- `figures/fig03_relative_error_boxplot.png/pdf` (aktualisiert)
- `figures/fig04_fd_step_study.png/pdf` (aktualisiert)
- `figures/fig05_error_by_scenario.png/pdf` (aktualisiert)
- `figures/fig06_gradient_magnitude_vs_relative_error_heatmaps.png/pdf` (aktualisiert)
- `figures/fig07_fd_vs_fd_step_stability.png/pdf` (aktualisiert)
- `figures/fig08_fd_step_study_detailed.png/pdf` (neu)
- `figures/fig09_fd_vs_fd_step_stability_detailed.png/pdf` (neu)

### Testergebnisse
- Ausgefuehrt:
  `.venv\Scripts\python.exe -m pytest tests\test_exp02_example_simple_gradients_outputs.py tests\test_exp02_plot_outputs.py -v`
  (`51 passed`)

### Ergebnisnotizen
- 48 Gradienten erzeugt und vollstaendig konvergiert (AD und FD). `fd_step = 1e-3`.
- Alle 12 Gruppen (Szenario x Eingangsparameter): valid=4/4.
- Median relativer AD-vs-FD-Fehler ueber alle validen Gradienten: ~1e-8.
- Maximaler relativer Fehler: 4.505e-05 (base: shunt_q_scale).
  Ursache: shunt_q_scale -> total_p_loss_mw hat einen sehr kleinen Gradienten
  (~7.4e-05 MW/dimensionless), was die relative Fehlermetrik gegenueber
  absolutem FD-Rauschen verstaerkt. Mit h=1e-3 liegt dieser Fall im FD-Plateau
  und ist deutlich besser als mit h=1e-4.
- Maximaler absoluter AD-vs-FD-Fehler: ~1.09e-9 (load_high: load_scale_mv_bus_2).
- Auffaelligste Input/Observable-Kombination: shunt_q_scale -> total_p_loss_mw
  (max_rel 4.505e-05); physikalisch erklaerbar, kein Implementierungsfehler.

#### Detaillierte StepStudy-Ergebnisse (3 repraesentative Gradienten)
- `load_scale_mv_bus_2 -> vm_mv_bus_2_pu`:
  Bestes h: 1e-3 (rel. Fehler: 4.19e-10). Schlechtestes h: 1e0 (4.54e-4).
  FD-Plateau stabil von h=1e-3 bis ~1e-4. Keine Instabilitaet bei kleinen h.
- `sgen_scale_static_generator -> p_slack_mw`:
  Bestes h: 1e-3 (rel. Fehler: 2.13e-11). Schlechtestes h: 1e-10 (2.18e-3).
  FD-Instabilitaet bei h <= 1e-8 sichtbar; bei h=1e-10 rel. Fehler 2.18e-3.
- `shunt_q_scale -> total_p_loss_mw`:
  Bestes h: 1e-2 (rel. Fehler: 5.83e-8). Schlechtestes h: 1e-10 (ca. 0.99,
  nahezu vollstaendige Auslöschung durch Rundungsfehler).
  FD-Instabilitaet ab h <= 1e-5 deutlich sichtbar. Bei h=1e-10 ist das FD-
  Ergebnis numerisch wertlos. Dies ist erwartet: der Gradient ist so klein
  (~7.4e-5 MW), dass 64-Bit-Auslöschung bei kleinen h dominiert.
  Interpretation: Der AD-Gradient ist korrekt; FD ist kein geeignetes
  Referenzverfahren fuer sehr kleine Schrittweiten bei kleinen Gradienten.

#### FD-vs-FD-Diagnose
- 30 benachbarte Schrittweitenpaare; alle 3 Gradienten x 10 Paare berechnet.
- `load_scale_mv_bus_2 -> vm_mv_bus_2_pu`: FD-vs-FD stabil bis h ~ 1e-6,
  dann leicht steigend. Keine dramatische Instabilitaet.
- `sgen_scale_static_generator -> p_slack_mw`: FD-vs-FD instabiler ab h <= 1e-7.
- `shunt_q_scale -> total_p_loss_mw`: FD-vs-FD zeigt stark steigende
  Instabilitaet ab h <= 1e-4 (fd_rel_change bis zu 1.0 bei h=1e-9/1e-10).
  Bestaetigt, dass FD-Referenz fuer diesen Fall ab h ~ 1e-5 unzuverlaessig wird.

### Bekannte Einschraenkungen
- Lokale Gradientenvalidierung bei theta0 = 1.0 (festem Betriebspunkt).
- Topologie statisch; keine Schaltungen oder Topologieaenderungen.
- Nur kontinuierliche Skalierungsparameter; kein vollstaendiger Jacobian.
- scope-matched: gen -> sgen(P, Q=0), keine Spannungsregelung.
- Keine Q-Limits, keine PV/PQ-Umschaltung, keine Controllerlogik.
- FD bei sehr kleinen h (shunt_q_case) numerisch instabil durch
  Gleitkomma-Ausloesung; AD-Gradient bleibt korrekt.

## 2026-05-19 - Scope-matched Open-Line-Policy fuer Experiment 1

### Kurzbeschreibung
- Der `scope_matched`-pandapower-Referenzlauf in Experiment 1 deaktiviert
  Leitungen hinter offenen Line-Switches nun vollstaendig mit
  `net.line.in_service = False`, bevor `runpp()` ausgefuehrt wird.
- Grund: diffpf entfernt diese Leitungen in der vereinfachten Topologie
  ebenfalls vollstaendig; pandapower wuerde sonst ueber Hilfsbusse kleine
  Ladeeffekte offener Leitungen beruecksichtigen.
- Der `original_pandapower`-Kontextvergleich bleibt unveraendert.
- Numerischer Kern, Solver, Residuum, Y-Bus und Trafo-Stempelung wurden nicht
  geaendert.

### Geaenderte Dateien
- `experiments/exp01_validate_example_simple.py` - Open-Line-Policy fuer
  `scope_matched`, erweiterte `structure_summary`- und Metadata-Dokumentation.
- `experiments/exp01_diagnose_residual_voltage_offset.py` - Diagnose auf die
  neue scope-matched Topologie umgestellt und Branch-Mapping fuer deaktivierte
  pandapower-Branches robust gemacht.
- `tests/test_exp01_example_simple_outputs.py` - Tests fuer die Open-Line-
  Policy und neue Strukturfelder ergaenzt.
- `tests/test_exp01_residual_voltage_offset_diagnostic.py` - bestehende
  Diagnose-Hilfsfunktionstests weiter genutzt.

### Aktualisierte Artefakte
- `experiments/results/exp01_example_simple_validation/*`
- `experiments/results/exp01_example_simple_validation/figures/*`
- `experiments/results/exp01_residual_voltage_offset_diagnostic/*`

### Ergebnisnotizen
- Betroffene Leitung in `example_simple()`: Line `2`.
- In `scope_matched` sehen pandapower und diffpf nun jeweils `3` aktive
  Leitungen.
- Max `max_vm_pu_abs_diff` sinkt von `2.350559e-05 p.u.` auf
  `4.773959e-14 p.u.`.
- Max `max_va_degree_abs_diff` sinkt von `2.527126e-04 deg` auf
  `2.188472e-12 deg`.
- Max `p_slack_mw_abs_diff` sinkt von `5.776102e-06 MW` auf
  `1.218972e-10 MW`.
- Max `total_p_loss_mw_abs_diff` sinkt von `5.775987e-06 MW` auf
  `1.106781e-12 MW`.
- Max `trafo_pl_mw_abs_diff` sinkt von `4.700529e-06 MW` auf
  `3.108624e-14 MW`.
- Die Restfehlerdiagnose bestaetigt die Ursache: Y-Bus-Maximaldifferenz von
  `4.178332e-03` auf `0.0`, `lines_only` von `4.178332e-03` auf `0.0`,
  diffpf-Residuum an der pandapower-Loesung von `4.290311e-03` auf
  `2.650863e-10`.

### Tests
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp01_example_simple_outputs.py -q`
  (`38 passed`)
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp01_residual_voltage_offset_diagnostic.py -q`
  (`4 passed`)
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp01_plot_outputs.py -q`
  (`9 passed`)
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_pandapower_adapter.py -q`
  (`15 passed, 1 xpassed`)
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_ybus.py -q`
  (`12 passed`)
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_trafo.py -q`
  (`12 passed`)
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe experiments/exp01_validate_example_simple.py`
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe experiments/plot_exp01_validation_figures.py`
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe experiments/exp01_diagnose_residual_voltage_offset.py`

## 2026-05-19 - Korrektur der Trafo-Magnetisierungsadmittanz im Pi-Stempel

### Kurzbeschreibung
- Die Transformator-Stempelung in `src/diffpf/core/ybus.py` wurde korrigiert.
- Das Modell verwendet weiterhin ein Pi-Ersatzschaltbild; es wurde nicht auf
  ein T-Ersatzschaltbild umgestellt.
- Die aus `pfe_kw` und `i0_percent` abgeleitete Leerlaufadmittanz wird nun als
  gesamte Magnetisierungsadmittanz `y_m = g_m - j*b_m` interpretiert und im
  Pi-Modell als `y_m/2` auf HV- und `y_m/2` auf LV-Klemme verteilt.
- Der HV-Selbstadmittanzterm nutzt nun `tap * conj(tap)`; die Off-Diagonalterme
  verwenden die konjugierten komplexen Tap-Faktoren analog zur
  `pandapower`-Pi-Referenz.
- `b_mag_pu` bleibt intern eine positive induktive Magnitude; das Vorzeichen
  wird erst im komplexen Y-Bus als `-j*b_mag_pu` gesetzt.

### Geaenderte Dateien
- `src/diffpf/core/ybus.py` - korrigierte Pi-Stempelung fuer
  Trafo-Magnetisierung, Tap-/Shift-Konjugation und Dokumentation.
- `src/diffpf/core/types.py` - Kommentar zur `b_mag_pu`-Vorzeichenkonvention
  praezisiert.
- `src/diffpf/io/pandapower_adapter.py` und `src/diffpf/io/parser.py` -
  Kommentare zur Speicherung von `b_mag_pu` als positiver Magnitude ergaenzt.
- `experiments/exp01_validate_example_simple.py` - Trafoflussauswertung,
  Metadata und Ergebnis-README an die korrigierte Stempelung angepasst;
  `p_slack_mw_abs_diff` in der Summary wird nun direkt aus der Slack-Tabelle
  uebernommen.
- `experiments/exp01_transformer_magnetization_ablation.py` - Interpretation
  des Ablationstests aktualisiert, damit der fruehere Diagnosebefund vom
  aktuellen korrigierten Hauptmodell getrennt wird.
- `experiments/plot_exp01_validation_figures.py` - Figure-README an den
  verschwundenen 14-kW-Offset angepasst.
- `tests/test_trafo.py` - Unit-Tests fuer halbierte Magnetisierung,
  reine Magnetisierung, positives `b_mag`-Vorzeichen und komplexe Tap-Terme
  erweitert.
- `docs/context/known_limitations.md` - Transformator-Paritaetsabschnitt auf
  den korrigierten Stand gebracht.
- `CHANGELOG.md` - dieser Eintrag.

### Aktualisierte Artefakte
- `experiments/results/exp01_example_simple_validation/validation_summary.csv/json`
- `experiments/results/exp01_example_simple_validation/bus_results.csv/json`
- `experiments/results/exp01_example_simple_validation/slack_results.csv/json`
- `experiments/results/exp01_example_simple_validation/line_flows.csv/json`
- `experiments/results/exp01_example_simple_validation/trafo_flows.csv/json`
- `experiments/results/exp01_example_simple_validation/losses.csv/json`
- `experiments/results/exp01_example_simple_validation/structure_summary.csv/json`
- `experiments/results/exp01_example_simple_validation/metadata.json`
- `experiments/results/exp01_example_simple_validation/README.md`
- `experiments/results/exp01_example_simple_validation/figures/*`
- `experiments/results/exp01_transformer_magnetization_ablation/*`

### Ergebnisnotizen
- Alle sieben `scope_matched`-Szenarien in Experiment 1 konvergieren weiterhin.
- Maximale Residualnorm der `scope_matched`-Szenarien: ca. `4.32e-11`.
- Max `max_vm_pu_abs_diff`: `2.350559e-05 p.u.`.
- Max `max_va_degree_abs_diff`: `2.527126e-04 deg`.
- Max `p_slack_mw_abs_diff`: `5.776102e-06 MW` = `0.005776 kW`.
- Mean `p_slack_mw_abs_diff`: `0.004602 kW`.
- Max `total_p_loss_mw_abs_diff`: `5.775987e-06 MW` = `0.005776 kW`.
- Mean `total_p_loss_mw_abs_diff`: `0.004602 kW`.
- Max `trafo_pl_mw_abs_diff`: `4.700529e-06 MW` = `0.004701 kW`.
- Mean `trafo_pl_mw_abs_diff`: `0.003871 kW`.
- Der zuvor dokumentierte ca. `14.36 kW`-Wirkleistungsoffset ist damit im
  aktuellen `scope_matched`-Hauptlauf verschwunden bzw. auf wenige Watt
  reduziert.

### Tests
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_trafo.py -q`
  (`12 passed`)
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp01_example_simple_outputs.py -q`
  (`34 passed`)
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_ybus.py tests/test_json_trafo_shunt.py tests/test_pandapower_adapter.py -q`
  (`38 passed, 1 xpassed`)
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp01_transformer_magnetization_ablation.py -q`
  (`9 passed`)
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe experiments/exp01_validate_example_simple.py`
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe experiments/plot_exp01_validation_figures.py`
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe experiments/exp01_transformer_magnetization_ablation.py`

### Bekannte Einschraenkungen
- Weiterhin Pi-Ersatzschaltbild; keine T-Modell-Umstellung.
- Keine Aenderung am Newton-Solver, Residuum oder an der impliziten
  Differentiation.
- Keine allgemeine Garantie vollstaendiger `pandapower`-Featureparitaet fuer
  alle Transformatorfaelle, Tap-Seiten, Controller oder Q-Limit-Logiken.

## 2026-05-17 - Experiment 4 Training-Loss-Visualisierung

### Kurzbeschreibung
- Eine neue reine Plot-Pipeline fuer Experiment 4 erzeugt berichtstaugliche
  Trainingskurven aus dem vorhandenen Artefakt `training_history.csv`.
- Fig. 1 visualisiert Train- und Validation-MSE ueber den aufgezeichneten
  Trainingsschritten.
- Da die Trainingshistorie MAE-Spalten enthaelt, erzeugt die Pipeline zusaetzlich
  Fig. 2 fuer Train- und Validation-MAE in MW.
- Dies ist eine reine Re-Visualisierung bestehender Exp.-4-Artefakte. Es wurde
  kein neues NN-Training gestartet und es wurden keine neuen Power-Flow-Solves,
  AD-Sensitivitaeten oder FD-Checks berechnet.

### Neue/geaenderte Dateien
- `experiments/plot_exp04_training_figures.py` - neues direkt ausfuehrbares
  Matplotlib/Pandas-Plot-Skript fuer die Exp.-4-Trainingshistorie.
- `tests/test_exp04_plot_outputs.py` - leichte Tests fuer Importierbarkeit,
  Spaltenerkennung, Validierung, Plot-Export und Scope ohne Training oder
  Power-Flow.
- `experiments/results/exp04_modular_upstream_nn_surrogate/figures/README.md` -
  Datenquelle, Kurvenbedeutung, Log-Skalierung, finale Werte und Scope
  dokumentiert.
- `CHANGELOG.md` - dieser Eintrag.

### Neue Artefakte
- `experiments/results/exp04_modular_upstream_nn_surrogate/figures/fig01_training_loss_curve.png`
- `experiments/results/exp04_modular_upstream_nn_surrogate/figures/fig01_training_loss_curve.pdf`
- `experiments/results/exp04_modular_upstream_nn_surrogate/figures/fig02_training_mae_curve.png`
- `experiments/results/exp04_modular_upstream_nn_surrogate/figures/fig02_training_mae_curve.pdf`

### Datenquelle
- `experiments/results/exp04_modular_upstream_nn_surrogate/training_history.csv`
  mit den Spalten `step`, `train_mse`, `val_mse`, `train_mae_mw`,
  `val_mae_mw` und `learning_rate`.

### Ergebnisnotizen
- Finale Train-MSE: ca. `0.001764`.
- Finale Validation-MSE: ca. `0.001756`.
- Finale Train-MAE: ca. `0.06648 MW`.
- Finale Validation-MAE: ca. `0.06663 MW`.
- MSE- und MAE-Figuren verwenden logarithmische y-Achsen, da alle geplotteten
  Werte positiv sind.

### Tests
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe experiments/plot_exp04_training_figures.py`
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp04_plot_outputs.py`
  (13 passed)
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp04_modular_surrogate_outputs.py`
  (9 passed)
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_pq_surrogate_model.py`
  (7 passed)

### Bekannte Einschraenkungen
- Die Figuren bewerten nur die bereits exportierte Trainingshistorie; sie
  starten kein Training neu und pruefen keine neuen Wetterpunkte.
- Die Visualisierung ist deskriptiv und ersetzt keine erneute
  Surrogatvalidierung, keinen Power-Flow-Modellvergleich und keine
  AD-vs-FD-Pruefung.
- Das NN bleibt das bestehende synthetische Distillation-Surrogat und kein
  Messdaten- oder allgemeines PV-Prognosemodell.

## 2026-05-17 - Experiment 4 grosses NN-Surrogat-Distillationstraining

### Kurzbeschreibung
- Experiment 4 wurde auf die robuste Hauptlauf-Konfiguration fuer das
  JAX-only NN-PV-Surrogat umgestellt.
- Die Default-Konfiguration verwendet nun `8192` Trainingspunkte, `2048`
  Validierungspunkte und einen separaten `2048`-Punkte-Evaluationssplit.
- Die Wetterbereiche bleiben unveraendert: `g_poa_wm2 = 0..1200 W/m^2`,
  `t_amb_c = -10..45 degC`, `wind_ms = 0.5..10 m/s`.
- Die Input-Normalisierung bleibt unveraendert:
  `(G - 600) / 600`, `(T_amb - 17.5) / 27.5`,
  `(wind - 5.25) / 4.75`.
- Das Trainingsziel bleibt die analytische PV-Wettermodell-Distillation
  `P_ref_mw / 2.0`. Das NN bleibt P-only; `Q = -0.25 * P` bleibt
  deterministisch gekoppelt.
- Der grosse Eval-Split dient nur der Surrogatfehler-Auswertung. Die kleine
  feste Wetterfallliste fuer Power-Flow-Vergleich, AD-vs-FD-Spotchecks und
  Sensitivitaetsmuster bleibt bewusst kompakt.

### Geaenderte Dateien
- `src/diffpf/models/pq_surrogate.py` - Default-Datensatzgroessen auf
  `8192/2048/2048` gesetzt und `eval_samples` in
  `SurrogateTrainingConfig` ergaenzt.
- `src/diffpf/models/__init__.py` - neue Default-Groessenkonstanten exportiert.
- `experiments/exp04_modular_upstream_nn_surrogate.py` - separater
  Eval-Split, Artefaktexport, Metadata, README und Run-Summary auf die neue
  Hauptlauf-Konfiguration erweitert.
- `tests/test_exp04_modular_surrogate_outputs.py` - Tests fuer die neue
  Default-Konfiguration und die kleine Smoke-Test-Konfiguration angepasst.
- `docs/context/experiment_04_nn_surrogate_plan.md` - Hauptlauf-Konfiguration
  und Trennung zwischen Eval-Split und kompakten Power-Flow-Vergleichsfaellen
  dokumentiert.
- `CHANGELOG.md` - dieser Eintrag.

### Aktualisierte Artefakte
- `experiments/results/exp04_modular_upstream_nn_surrogate/metadata.json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/README.md`
- `experiments/results/exp04_modular_upstream_nn_surrogate/training_dataset_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/training_history.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/surrogate_error_table.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/model_comparison.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/coupling_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/gradient_success_table.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/sensitivity_pattern_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/run_summary.csv/json`

### Ergebnisnotizen
- `training_dataset_summary.csv` dokumentiert `train=8192`, `val=2048`,
  `eval=2048`.
- `surrogate_error_table.csv` enthaelt `12288` Zeilen fuer Train-, Val- und
  Eval-Split zusammen.
- Der kompakte Power-Flow-Modellvergleich bleibt bei `18` Loesungen pro
  Upstream-Modell; alle drei Modelle konvergieren mit `18/18`.
- Finale Trainingshistorie: `val_mae_mw` ca. `0.0666 MW`.

### Tests
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe experiments/exp04_modular_upstream_nn_surrogate.py`
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp04_modular_surrogate_outputs.py -q`
  (9 passed)
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_pq_surrogate_model.py -q`
  (7 passed)
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_pv_model.py -q`
  (16 passed)
- Optional ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp03_cross_domain_outputs.py -q`
  (27 passed)

### Bekannte Einschraenkungen
- Das NN bleibt ein synthetisches Distillation-Surrogat, kein Messdaten- oder
  allgemeines PV-Prognosemodell.
- P-only-MLP: Blindleistung wird nicht gelernt, sondern bleibt fest
  `Q = -0.25 * P`.
- Keine Aenderung am AC-Power-Flow-Kern, PV-Modell, Solver, Residuen, Y-Bus,
  impliziter Differentiation, Observables oder pandapower-Adapter.
- Keine PV-Bus-Spannungsregelung, keine Q-Limits, keine PV-PQ-Umschaltung und
  keine Controllerlogik.

## 2026-05-17 - Experiment 3 Sensitivitaets-Heatmaps im G-T-Gitter

### Kurzbeschreibung
- Die bestehende Exp.-3-Plot-Pipeline wurde um zwei diskrete
  Sensitivitaets-Heatmaps fuer das vorhandene 2D-Wettergitter ergaenzt.
- Neue Fig. 6 zeigt `G x T -> dP_slack/dg_poa_wm2` fuer das
  `base`-Szenario und `p_slack_mw`.
- Neue Fig. 7 zeigt `G x T -> dP_slack/dt_amb_c` fuer das
  `base`-Szenario und `p_slack_mw`.
- Dies ist eine reine Re-Visualisierung vorhandener Exp.-3-Artefakte aus
  `sensitivity_table.csv`. Es wurden keine neuen Power-Flow-Solves, keine
  neuen AD-Sensitivitaeten und keine neuen FD-Checks gestartet.
- Das PV-Modell, der numerische Kern, Solver, Compile-Schicht, Residuen,
  Y-Bus und pandapower-Adapter bleiben unveraendert.

### Geaenderte Dateien
- `experiments/plot_exp03_figures.py` - neue Pandas-basierte Pivot-Hilfen,
  Einheitentransformationen und Fig.-6/Fig.-7-Heatmap-Exports ergaenzt.
- `tests/test_exp03_plot_outputs.py` - Tests fuer Importierbarkeit,
  Filter-/Pivot-Logik, Einheitentransformationen und neue Plot-Artefakte
  erweitert.
- `experiments/results/exp03_cross_domain_pv_weather/figures/README.md` -
  Fig. 6 und Fig. 7 mit Datenquelle, Filtern, Einheiten und Interpretation
  dokumentiert.
- `CHANGELOG.md` - dieser Eintrag.

### Neue Artefakte
- `experiments/results/exp03_cross_domain_pv_weather/figures/fig06_heatmap_g_t_sensitivity_p_slack_wrt_g.png`
- `experiments/results/exp03_cross_domain_pv_weather/figures/fig06_heatmap_g_t_sensitivity_p_slack_wrt_g.pdf`
- `experiments/results/exp03_cross_domain_pv_weather/figures/fig07_heatmap_g_t_sensitivity_p_slack_wrt_t_amb.png`
- `experiments/results/exp03_cross_domain_pv_weather/figures/fig07_heatmap_g_t_sensitivity_p_slack_wrt_t_amb.pdf`

### Ergebnisnotizen
- `dP_slack/dG` wird in kW pro 100 W/m^2 dargestellt und ist im betrachteten
  5x5-Gitter durchgehend negativ, ca. `-155` bis `-210` kW pro 100 W/m^2.
- `dP_slack/dT_amb` wird in kW/degC dargestellt und ist im betrachteten
  5x5-Gitter durchgehend positiv, ca. `1.6` bis `8.0` kW/degC.
- Die Vorzeichen sind plausibel: mehr Einstrahlung erhoeht die PV-Einspeisung
  und macht `P_slack` negativer; hoehere Umgebungstemperatur senkt die
  PV-Leistung und macht `P_slack` weniger negativ.

### Tests
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe experiments/plot_exp03_figures.py`
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp03_plot_outputs.py`
  (19 passed)

### Bekannte Einschraenkungen
- Reine Re-Visualisierung vorhandener Sensitivitaetsdaten; keine neuen
  numerischen Laeufe und keine neuen Szenarien.
- Die Heatmaps zeigen lokale Ableitungen, keine globalen Differenzen ueber das
  Wettergitter.
- Keine Aenderung der physikalischen Vorzeichenkonvention.
- Alle bekannten Exp.-3-Modellgrenzen bleiben bestehen: PQ-PV-Einspeisung,
  keine PV-Bus-Regelung, keine Q-Limits, keine PV-PQ-Umschaltung und keine
  Controllerlogik.

## 2026-05-17 - Experiment 2 FD-vs-FD-Schrittweitenstabilitaet

### Kurzbeschreibung
- Die bestehende Exp.-2-Plot- und Auswertungspipeline wurde um eine
  FD-vs-FD-Stabilitaetsdiagnose fuer die vorhandene FD-Schrittweitenstudie
  ergaenzt.
- Die neue Analyse vergleicht benachbarte FD-Gradientenpaare, z. B. `FD(h)`
  gegen `FD(h/10)`, und bewertet damit die Stabilitaet der
  Finite-Difference-Referenz selbst.
- Dies ist eine reine Re-Analyse vorhandener Exp.-2-Artefakte. Es wurden keine
  neuen Power-Flow-Solves, keine neuen AD-Gradienten und keine neuen
  Finite-Difference-Laeufe gestartet.
- Der numerische Kern bleibt unveraendert; keine Aenderungen an `core/`,
  `solver/`, `compile/`, Residuen, Y-Bus oder pandapower-Adapter.

### Geaenderte Dateien
- `experiments/plot_exp02_gradient_figures.py` - neue Funktion
  `compute_fd_vs_fd_step_stability(...)`, CSV/JSON-Export und Fig.-7-Plot
  ergaenzt.
- `tests/test_exp02_plot_outputs.py` - Tests fuer Funktion, Tabellenform,
  Plausibilitaet, Exporte und Regression der bestehenden Figuren erweitert.
- `experiments/results/exp02_example_simple_gradients/figures/README.md` -
  Fig. 7 mit Zweck, Datenquelle, Interpretation und Grenzen dokumentiert.
- `CHANGELOG.md` - dieser Eintrag.

### Neue Artefakte
- `experiments/results/exp02_example_simple_gradients/figures/fd_vs_fd_step_stability.csv`
- `experiments/results/exp02_example_simple_gradients/figures/fd_vs_fd_step_stability.json`
- `experiments/results/exp02_example_simple_gradients/figures/fig07_fd_vs_fd_step_stability.png`
- `experiments/results/exp02_example_simple_gradients/figures/fig07_fd_vs_fd_step_stability.pdf`

### Ergebnisnotizen
- Die FD-vs-FD-Tabelle enthaelt 12 Zeilen: 3 ausgewaehlte Gradienten mit je 4
  benachbarten Schrittweitenpaaren.
- `load_scale_mv_bus_2 -> vm_mv_bus_2_pu` zeigt ein stabiles FD-Plateau im
  mittleren Schrittweitenbereich; das kleinste Paar bleibt im Bereich weniger
  `1e-9` relativer FD-Aenderung.
- `sgen_scale_static_generator -> p_slack_mw` bleibt ueber mittlere
  Schrittweiten sehr stabil und zeigt erst beim kleinsten Paar eine hoehere,
  aber weiterhin kleine relative FD-Aenderung.
- `shunt_q_scale -> total_p_loss_mw` zeigt die erwartete staerkere
  Instabilitaet bei kleinen Schrittweiten: `fd_rel_change` steigt bis auf ca.
  `4.1e-3` fuer `1e-05 -> 1e-06`.

### Tests
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe experiments/plot_exp02_gradient_figures.py`
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp02_plot_outputs.py`
  (24 passed)

### Bekannte Einschraenkungen
- FD-vs-FD ist eine Zusatzdiagnose und ersetzt nicht den bestehenden
  AD-vs-FD-Hauptvergleich.
- Die Diagnose nutzt ausschliesslich die drei bereits ausgewaehlten Gradienten
  der vorhandenen `fd_step_study.csv`; sie ist keine vollstaendige neue
  Gradientenvalidierung.
- Alle bekannten Modellgrenzen von Exp. 2 bleiben bestehen, insbesondere
  scope-matched `gen -> sgen(P, Q=0)`, keine Q-Limits, keine
  PV-PQ-Umschaltung und keine Controllerlogik.

## 2026-05-16 - Experiment 1 Transformer-Magnetisierungsablation

### Kurzbeschreibung
- Neuer diagnostischer Ablationstest fuer Experiment 1 ergaenzt. Der Test
  prueft, ob der systematische ca. 14-kW-Wirkleistungsoffset im
  `scope_matched`-Vergleich von `pandapower.networks.example_simple()` durch
  Trafo-Magnetisierung bzw. Eisenverluste erklaert wird.
- Dafuer werden alle sieben Exp.-1-Szenarien einmal im bisherigen
  `baseline`-Scope und einmal mit `pfe_kw = 0.0` sowie
  `i0_percent = 0.0` am Trafo `110kV/20kV transformer` gerechnet.
- Der numerische Kern bleibt unveraendert; Y-Bus, Residuen, Solver,
  Trafo-Stempelung und Observables wurden nicht geaendert.

### Neue Dateien
- `experiments/exp01_transformer_magnetization_ablation.py` - direkt
  ausfuehrbares Diagnose-Skript mit Variantenbildung, Summary,
  Hypothesencheck, Artefaktexport und optionaler Balkengrafik.
- `tests/test_exp01_transformer_magnetization_ablation.py` - Tests fuer
  Trafo-Finder, Ablationsfunktion, Summary-Logik, Hypothesencheck und Export.

### Neue Artefakte
- `experiments/results/exp01_transformer_magnetization_ablation/ablation_results.csv/json`
- `experiments/results/exp01_transformer_magnetization_ablation/ablation_summary.csv/json`
- `experiments/results/exp01_transformer_magnetization_ablation/hypothesis_check.json`
- `experiments/results/exp01_transformer_magnetization_ablation/metadata.json`
- `experiments/results/exp01_transformer_magnetization_ablation/README.md`
- `experiments/results/exp01_transformer_magnetization_ablation/figures/fig01_p_offset_baseline_vs_ablation.png/pdf`

### Ergebnisnotizen
- Baseline mean `p_slack_mw_abs_diff`: `14.364137 kW`.
- Ablated mean `p_slack_mw_abs_diff`: `0.004896 kW`.
- Baseline mean `total_p_loss_mw_abs_diff`: `14.364137 kW`.
- Ablated mean `total_p_loss_mw_abs_diff`: `0.004896 kW`.
- Baseline mean `trafo_pl_mw_abs_diff`: `14.374564 kW`.
- Ablated mean `trafo_pl_mw_abs_diff`: `0.004170 kW`.
- Reduktionsfaktoren: `p_slack = 2933.70`,
  `total_p_loss = 2933.77`, `trafo_pl = 3447.22`.
- Die Hypothese wird durch den Ablationstest unterstuetzt.

### Tests
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe experiments/exp01_transformer_magnetization_ablation.py`
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp01_transformer_magnetization_ablation.py -q`
  (9 passed)

### Bekannte Einschraenkungen
- Reine Diagnose; keine Korrektur der Trafo-Stempelung und keine
  vollstaendige pandapower-Kompatibilitaet.
- Der Test isoliert nur die Wirkung von `pfe_kw` und `i0_percent`.
- Keine Aenderungen an `core/`, `solver/`, `compile/`, Y-Bus oder Residuen.

## 2026-05-14 - Experiment 5 Plot-Pipeline aus vorhandenen Artefakten

### Kurzbeschreibung
- Neue Plot-Pipeline fuer Experiment 5 ergaenzt. Das Skript erzeugt vier
  berichtstaugliche Abbildungen ausschliesslich aus bestehenden CSV-Artefakten
  von Exp. 5a und Exp. 5b.
- Es werden keine neuen Power-Flow-Solves, keine Sensitivitaeten, keine
  Grid-Search und keine Curtailment-Optimierung gestartet.

### Neue Dateien
- `experiments/plot_exp05_figures.py` - liest vorhandene Exp.-5-Artefakte,
  validiert die benoetigten Spalten und erzeugt PNG/PDF-Figuren.
- `tests/test_exp05_plot_outputs.py` - isolierte Tests fuer Spaltenvalidierung,
  Grid-Bestimmung und Figurenerzeugung mit Dummy-Artefakten.

### Neue Artefakte
- `experiments/results/exp05_figures/fig51_screening_export_overview.png/pdf`
- `experiments/results/exp05_figures/fig53_export_before_after_reference.png/pdf`
- `experiments/results/exp05_figures/fig54_grid_reference_export_vs_curtailment.png/pdf`
- `experiments/results/exp05_figures/fig55_optimization_trace_export_and_curtailment.png/pdf`
- `experiments/results/exp05_figures/README.md`
- `experiments/results/exp05_figures/figure_metadata.json`

### Datenquellen
- `experiments/results/exp05a_network_screening/screening_results.csv`
- `experiments/results/exp05a_network_screening/selected_realistic_case.csv`
- `experiments/results/exp05b_optimize_pv_curtailment/selected_case_baseline.csv`
- `experiments/results/exp05b_optimize_pv_curtailment/final_solution.csv`
- `experiments/results/exp05b_optimize_pv_curtailment/grid_reference.csv`
- `experiments/results/exp05b_optimize_pv_curtailment/optimization_trace.csv`
- `experiments/results/exp05b_optimize_pv_curtailment/constraint_diagnostics.csv`

### Tests
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe experiments/plot_exp05_figures.py`
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp05_plot_outputs.py -q`
  (5 passed)

### Bekannte Einschraenkungen
- Reine Reporting-Ergaenzung; keine methodische Aenderung am Modell.
- Die 7.0-MW-Linie bleibt ein demonstratorinterner Zielwert, keine normative
  Netzcode-Grenze.
- Keine thermische Betriebsmittelbewertung; alle Einschraenkungen aus Exp. 5a
  und Exp. 5b bleiben bestehen.

## 2026-05-13 - Experiment 5a Auswahlfall und Experiment 5b PV-Curtailment

### Kurzbeschreibung
- Experiment 5a wurde um den separaten realistischeren Sommer-Hoch-PV-Fall
  `selected_realistic_load0p4_g1200_t30` ergaenzt. Der urspruengliche
  48-Fall-Screeningumfang bleibt unveraendert.
- Experiment 5b implementiert eine gradientenbasierte Optimierung des
  PV-Curtailment-Faktors fuer genau diesen ausgewaehlten Betriebspunkt.

### Neue und geaenderte Dateien
- `experiments/exp05a_network_screening.py` - separater 30-C-Auswahlfall,
  no-PV-Deltas und lokale Sensitivitaeten fuer den Auswahlfall.
- `experiments/exp05b_optimize_pv_curtailment.py` - neues Experiment mit
  Sigmoid-Parametrisierung, glatter Export-Penalty, lokalem Adam-Loop,
  Feasibility-Diagnostik und 1D-Grid-Referenz.
- `tests/test_exp05a_network_screening_outputs.py` - Tests fuer den separaten
  Auswahlfall und dessen Artefakte.
- `tests/test_exp05b_optimize_pv_curtailment_outputs.py` - Schema- und
  Smoke-Tests fuer Exp. 5b.
- `docs/context/experiment_plan.md`, `docs/context/validation_status.md`,
  `docs/context/known_limitations.md` - knappe Einordnung von Exp. 5a/5b.

### Neue Artefakte
- Exp. 5a:
  - `experiments/results/exp05a_network_screening/selected_realistic_case.csv/json`
  - `experiments/results/exp05a_network_screening/selected_realistic_case_sensitivity.csv/json`
- Exp. 5b:
  - `experiments/results/exp05b_optimize_pv_curtailment/selected_case_baseline.csv/json`
  - `experiments/results/exp05b_optimize_pv_curtailment/optimization_trace.csv/json`
  - `experiments/results/exp05b_optimize_pv_curtailment/final_solution.csv/json`
  - `experiments/results/exp05b_optimize_pv_curtailment/grid_reference.csv/json`
  - `experiments/results/exp05b_optimize_pv_curtailment/constraint_diagnostics.csv/json`
  - `experiments/results/exp05b_optimize_pv_curtailment/run_summary.csv/json`
  - `experiments/results/exp05b_optimize_pv_curtailment/metadata.json`
  - `experiments/results/exp05b_optimize_pv_curtailment/README.md`

### Ergebnisnotizen
- Der ausgewaehlte 30-C-Fall verletzt bei voller PV den demonstratorinternen
  Exportzielwert: `p_export_mw = 7.599971`.
- Bei `c = 0.0` ist die Grenze erreichbar: `p_export_mw = 5.462384`.
- Der Exp.-5b-Optimizer nutzt nun eine Target-Tracking-Variante mit
  `p_export_target_mw = 6.99`, `beta = 300` und explizitem Export von
  `hard_export_violation_mw` sowie `soft_export_violation_mw`.
- Der Exp.-5b-Optimizer landet bei `c = 0.714203`, also
  `71.4203 %` PV-Nutzung und `28.5797 %` Abregelung.
- Finaler Export: `6.990006 MW`, Export-Margin: `0.009994 MW`.
- Grid-Referenz: groesster zulaessiger Grid-Wert `c = 0.718`.

### Tests
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe experiments/exp05a_network_screening.py`
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe experiments/exp05b_optimize_pv_curtailment.py`
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp05a_network_screening_outputs.py -q`
  (15 passed)
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp05b_optimize_pv_curtailment_outputs.py -q`
  (9 passed)
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_pv_model.py -q`
  (16 passed)
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp03_cross_domain_outputs.py -q`
  (27 passed)
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp04_modular_surrogate_outputs.py -q`
  (8 passed)

### Bekannte Einschraenkungen
- Exp. 5b optimiert nur einen ausgewaehlten Betriebspunkt, keine Szenariofamilie.
- Der Exportgrenzwert `7.0 MW` ist ein demonstratorinterner Zielwert, keine
  normative Netzcode-Grenze.
- PV bleibt eine wetterabhaengige PQ-Einspeisung; keine PV-Bus-Regelung, keine
  Q-Limits, keine PV-PQ-Umschaltung und keine Controllerlogik.
- Keine thermische Betriebsmittelbewertung; Trafo- und Leitungswerte bleiben
  diagnostische Proxies.
- Keine Aenderungen an `core/`, `solver/`, `compile/`, Y-Bus oder Residuen.

## 2026-05-13 - Experiment 5a Netzscreening fuer PV-Curtailment-Vorbereitung

### Neue Dateien
- `experiments/exp05a_network_screening.py` - reduziertes Forward-Screening
  auf `pandapower.networks.example_simple()` im scope-matched Modell. Das
  `sgen "static generator"` am Bus `"MV Bus 2"` wird durch das bestehende
  JAX-kompatible PV-Wettermodell ersetzt. Es werden 48 PV-Screeningfaelle und
  4 no-PV-Referenzfaelle geloest, demonstratorinterne Stressindikatoren
  berechnet, die Top-20-Faelle selektiert und nur fuer diese Top-20 lokale
  AD-Sensitivitaeten nach `curtailment_factor` berechnet.
- `tests/test_exp05a_network_screening_outputs.py` - leichte Schema-,
  Scoring- und Exporttests fuer die neuen Experimentartefakte.

### Neue Artefakte
- `experiments/results/exp05a_network_screening/screening_results.csv/json`
- `experiments/results/exp05a_network_screening/top_critical_cases.csv/json`
- `experiments/results/exp05a_network_screening/sensitivity_top20.csv/json`
- `experiments/results/exp05a_network_screening/branch_flows.csv/json`
- `experiments/results/exp05a_network_screening/run_summary.csv/json`
- `experiments/results/exp05a_network_screening/selected_realistic_case.csv/json`
- `experiments/results/exp05a_network_screening/selected_realistic_case_sensitivity.csv/json`
- `experiments/results/exp05a_network_screening/metadata.json`
- `experiments/results/exp05a_network_screening/README.md`

### Ergebnisumfang
- Forward-Faelle: 52 = 48 Screeningfaelle + 4 no-PV-Referenzen.
- Konvergenz: 52/52 Forward-Faelle konvergiert.
- Top-Cases: 20.
- Sensitivitaetszeilen: 160 = 20 Top-Cases x 8 Observables.
- AD-Sensitivitaeten: 0 fehlgeschlagene Zeilen im erzeugten Artefakt.

### Tests
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp05a_network_screening_outputs.py -q`
  (15 passed)
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe experiments/exp05a_network_screening.py`

### Grenzen
- Keine PV-Curtailment-Optimierung; nur Screening und lokale Sensitivitaeten.
- Kritikalitaetsflags sind demonstratorinterne Stressindikatoren, keine
  normativen Netzcode- oder Betriebsmittelgrenzverletzungen.
- Keine Controllerlogik, keine Q-Limits, keine PV-PQ-Umschaltung und keine
  neue PV-Bus-Regelung.
- Leitungsfluesse werden nur als Scheinleistungs-Proxies exportiert; es werden
  keine Leitungsauslastungen in Prozent behauptet.
- Keine Aenderungen an `core/`, `solver/`, `compile/`, Y-Bus, Residuen oder
  physikalischer Solverformulierung.

## 2026-05-07 - Experiment 3 Einstrahlungs-Sweep ergaenzt

### Geaenderte Dateien
- `experiments/exp03_cross_domain_pv_weather.py` - neues Wetterdesign
  `sweep_g_1d` ergaenzt: `g_poa_wm2 = {200, 400, 600, 800, 1000}` bei
  `t_amb_c = 25.0` und `wind_ms = 2.0`.
- `experiments/plot_exp03_figures.py` - neue Fig. 4 fuer den
  Einstrahlungs-Sweep der Slack-Wirkleistung und neue Fig. 5 fuer die lokale
  AD-Sensitivitaet `dP_slack/dG_poa` ergaenzt.
- `tests/test_exp03_cross_domain_outputs.py` - Wetterfalltyp,
  Szenariozaehler, Metadaten und neue Sweep-Schemafaelle abgesichert.
- `tests/test_exp03_plot_outputs.py` - Filter, Einheitenumrechnung,
  Plot-Export und Achsenlabels fuer den Einstrahlungs-Sweep abgesichert.
- `experiments/results/exp03_cross_domain_pv_weather/README.md` und
  `metadata.json` - neues Wetterdesign und Artefaktgroessen dokumentiert.
- `experiments/results/exp03_cross_domain_pv_weather/figures/README.md` -
  Fig. 4 und Fig. 5 dokumentiert.

### Neue Wetterfallart
- `sweep_g_1d`: 5 Einstrahlungswerte bei fester Umgebungstemperatur und festem
  Wind.
- Forward-Solves: 108
- Sensitivitaetszeilen: 1296

### Neue Artefakte
- `experiments/results/exp03_cross_domain_pv_weather/figures/fig04_g_sweep_p_slack.png/pdf`
- `experiments/results/exp03_cross_domain_pv_weather/figures/fig05_sensitivity_p_slack_vs_g_poa.png/pdf`

### Tests
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe experiments/exp03_cross_domain_pv_weather.py`
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe experiments/plot_exp03_figures.py`
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp03_cross_domain_outputs.py`
  (27 passed)
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp03_plot_outputs.py`
  (15 passed)
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_pv_model.py`
  (16 passed)

### Grenzen
- Keine Aenderung am PV-Modell, am Power-Flow-Kern, Solver, Residuen, Y-Bus,
  Compiler oder pandapower-Adapter.
- Keine neue vollstaendige FD-Validierung; die bestehenden Spotchecks bleiben
  kompakt.
- `alpha` und `kappa` bleiben feste Konstanten.

## 2026-05-07 - Experiment 2 Gradientengroessen-vs-Fehler-Heatmaps

### Geaenderte Dateien
- `experiments/plot_exp02_gradient_figures.py` - Aggregation der
  Gradientengroessen und relativen AD-vs-FD-Fehler je
  Input-/Output-Kombination ergaenzt; neue Fig. 6 mit zwei diskreten Heatmaps
  fuer `log10(median |AD gradient|)` und `log10(max relative error)` erzeugt.
- `tests/test_exp02_plot_outputs.py` - Tests fuer Aggregation, robuste
  Log-Floors, Fig.-6-Export, Colorbar-Labels und README-Abschnitt ergaenzt.
- `experiments/results/exp02_example_simple_gradients/figures/README.md` -
  Fig.-6-Datenquelle, Interpretation und Grenzen dokumentiert.

### Neue Artefakte
- `experiments/results/exp02_example_simple_gradients/figures/gradient_magnitude_vs_error_summary.csv`
- `experiments/results/exp02_example_simple_gradients/figures/gradient_magnitude_vs_error_summary.json`
- `experiments/results/exp02_example_simple_gradients/figures/fig06_gradient_magnitude_vs_relative_error_heatmaps.png`
- `experiments/results/exp02_example_simple_gradients/figures/fig06_gradient_magnitude_vs_relative_error_heatmaps.pdf`

### Tests
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe experiments/plot_exp02_gradient_figures.py`
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp02_plot_outputs.py`
  (19 passed)
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp02_example_simple_gradients_outputs.py`
  (8 passed)

### Grenzen
- Reine Re-Visualisierung vorhandener CSV-Artefakte.
- Keine neuen Power-Flow-Solves, keine neuen AD-Gradienten, keine neuen
  Finite-Difference-Laeufe und keine Aenderung an Ergebnisdaten oder
  numerischem Kern.

## 2026-05-07 - Experiment 2 Heatmap an Experiment-3-Darstellung angeglichen

### Geaenderte Dateien
- `experiments/plot_exp02_gradient_figures.py` - Fig. 2 wird weiterhin aus
  `gradient_table.csv` aggregiert, aber nun als diskrete Matplotlib-Heatmap mit
  sichtbaren Zellgrenzen, klarem Titel, stabilerem Layout,
  Achsenbeschriftungen und gut lesbarer Colorbar erzeugt.
- `tests/test_exp02_plot_outputs.py` - Tests fuer Titel, Achsenlabels,
  Colorbar-Beschriftung und Heatmap-Dateiexport ergaenzt.
- `experiments/results/exp02_example_simple_gradients/figures/README.md` -
  Fig.-2-Beschreibung um log10-Darstellung, getrennte Zellen und den
  Artefakt-only-Charakter ergaenzt.

### Aktualisierte Artefakte
- `experiments/results/exp02_example_simple_gradients/figures/fig02_gradient_error_heatmap.png`
- `experiments/results/exp02_example_simple_gradients/figures/fig02_gradient_error_heatmap.pdf`

### Tests
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe experiments/plot_exp02_gradient_figures.py`
- Ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp02_plot_outputs.py`
  (12 passed)
- Optional ausgefuehrt:
  `.venv\\Scripts\\python.exe -m pytest tests/test_exp03_plot_outputs.py`
  (8 passed, 1 bestehender Exp.-3-Fehler wegen fehlendem `padded_limits` in
  `experiments.plot_exp03_figures`)

### Grenzen
- Reine Re-Visualisierung vorhandener Exp.-2-Artefakte.
- Keine neuen Power-Flow-Solves, keine neuen AD-Gradienten, keine neuen
  Finite-Difference-Laeufe und keine Aenderung an Ergebnisdaten oder
  numerischer Kernlogik.

## 2026-05-07 - Experiment 1 Visualisierungen aus bestehenden Artefakten

### Neue Dateien
- `experiments/plot_exp01_validation_figures.py` - direkt ausfuehrbare
  Plot-Pipeline fuer scope-matched Validierungsfehler aus
  `validation_summary.csv`; erzeugt Long-Table, Stabilitaetszusammenfassung,
  Szenario-Plot und Boxplot-Darstellung ohne neue Power-Flow-Solves.
- `tests/test_exp01_plot_outputs.py` - leichte Tests fuer Importierbarkeit,
  scope-matched-Filter, Tidy-Long-Table, Einheitentransformation,
  Stabilitaetsstatistik, robuste CV-Berechnung, Plot-Export sowie
  README-/CSV-/JSON-Export.

### Erzeugte Artefakte
- `experiments/results/exp01_example_simple_validation/figures/fig01_scope_matched_error_by_scenario.png/pdf`
- `experiments/results/exp01_example_simple_validation/figures/fig02_scope_matched_error_boxplots.png/pdf`
- `experiments/results/exp01_example_simple_validation/figures/scope_matched_error_long_table.csv/json`
- `experiments/results/exp01_example_simple_validation/figures/scope_matched_error_stability_summary.csv/json`
- `experiments/results/exp01_example_simple_validation/figures/README.md`

### Tests
- Ausgefuehrt: `python experiments/plot_exp01_validation_figures.py`
- Ausgefuehrt: `python -m pytest tests/test_exp01_plot_outputs.py`
  (9 passed)
- Ausgefuehrt: `python -m pytest tests/test_exp01_example_simple_outputs.py`
  (34 passed)
- Zusaetzlicher Plot-Testlauf:
  `python -m pytest tests/test_exp02_plot_outputs.py tests/test_exp03_plot_outputs.py`
  (18 passed, 1 bestehender Exp.-3-Fehler wegen fehlendem
  `padded_limits` in `experiments.plot_exp03_figures`)

### Grenzen
- Reine deskriptive Auswertung vorhandener Exp.-1-Artefakte.
- Keine neuen pandapower- oder Power-Flow-Laeufe, keine neue
  Validierungslogik, keine Gradientenberechnung und keine Aenderung am
  numerischen JAX-Kern.
- Die Figuren stuetzen die Interpretation eines systematischen
  modellstrukturellen Offsets, sind aber kein statistischer Beweis.

## 2026-05-05 - Experiment 4: Modulare Upstream-Kopplung mit NN-PQ-Surrogat

### Neue Dateien
- `src/diffpf/models/pq_surrogate.py` - kleines JAX-only MLP als P-only
  Upstream-Surrogat ohne Equinox/Flax/Optax; `Q = kappa * P` bleibt fest.
- `experiments/exp04_modular_upstream_nn_surrogate.py` - direkt ausfuehrbares
  Experiment 4 mit synthetischer Distillation des analytischen PV-Wettermodells,
  Modellvergleich, Kopplungsnachweis, AD-vs-FD-Spotchecks und
  Sensitivitaetsmustervergleich.
- `tests/test_pq_surrogate_model.py` - Modelltests fuer deterministische
  Initialisierung, Shape/Finite-Werte, Q/P-Verhaeltnis, JIT, Gradienten,
  Parameterzahl und pandapower-freies Modellmodul.
- `tests/test_exp04_modular_surrogate_outputs.py` - leichte Schema- und
  Exporttests fuer die Exp.-4-Artefakte sowie ein Mini-Training-Smoke-Test.
- `docs/context/experiment_04_nn_surrogate_plan.md` - wissenschaftlicher Plan
  fuer Ziel, Architektur, Datensatz, Kopplung, Artefakte und Grenzen.

### Geaenderte Dateien
- `src/diffpf/models/__init__.py` - Surrogatmodell-API exportiert.

### Artefakte
- `experiments/results/exp04_modular_upstream_nn_surrogate/metadata.json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/README.md`
- `experiments/results/exp04_modular_upstream_nn_surrogate/training_dataset_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/training_history.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/surrogate_error_table.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/model_comparison.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/coupling_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/gradient_success_table.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/sensitivity_pattern_summary.csv/json`
- `experiments/results/exp04_modular_upstream_nn_surrogate/run_summary.csv/json`

### Tests
- Vorgesehen: `python -m pytest tests/test_pq_surrogate_model.py tests/test_exp04_modular_surrogate_outputs.py`
- Vorgesehen: `python -m pytest tests/test_pv_model.py tests/test_exp03_cross_domain_outputs.py`

### Modellannahmen und Grenzen
- Das NN ist ein kleines Distillation-Surrogat fuer Modularitaet, kein
  Messdaten-Prognosemodell.
- Hauptpfad ist P-only; Blindleistung folgt deterministisch aus
  `Q = -0.25 * P`.
- Alle Upstream-Modelle nutzen denselben P/Q-Injektionsadapter und denselben
  unveraenderten Power-Flow-Kern.
- Keine echte PV-Bus-Spannungsregelung, keine Q-Limits, keine
  PV-PQ-Umschaltung und keine Controllerlogik.

## 2026-05-04 - Experiment 3 Plot-Verbesserungen

### Geaenderte Dateien
- `experiments/plot_exp03_figures.py` - berichtstauglichere Replots der drei
  bestehenden Exp.-3-Figuren aus den vorhandenen CSV-Artefakten: feste
  Sweep-/Grid-Ticks, klarere Achsenlabels, Slack-Vorzeichenhinweis,
  diskrete 5x5-Heatmap-Zellen und Fig.-3-Sensitivitaet in kW/degC.
  Die Heatmap verwendet nur noch einen Haupttitel; Fig. 3 ist wieder der
  Temperatursweep-Linienplot je Szenario.
- `tests/test_exp03_plot_outputs.py` - leichte Tests fuer die festen
  Darstellungs-Ticks, die Umrechnung der Fig.-3-Sensitivitaeten von
  MW/degC nach kW/degC.
- `experiments/results/exp03_cross_domain_pv_weather/figures/README.md` -
  aktualisierte Datenquellen, Filter, Vorzeichenkonvention und
  Einheitentransformation dokumentiert.

### Grenzen
- Es wurden keine neuen Power-Flow-Solves, Szenarien, AD-Sensitivitaeten oder
  Finite-Difference-Berechnungen gestartet.
- Die Plot-Erzeugung bleibt reine Auswertung von `scenario_grid.csv` und
  `sensitivity_table.csv`; die Ergebnisdaten selbst bleiben unveraendert.

## 2026-05-01 - Experiment 2 Visualisierungen aus bestehenden Artefakten

### Neue Dateien
- `experiments/plot_exp02_gradient_figures.py` - erzeugt fuenf
  publikationsnahe Matplotlib-Abbildungen direkt aus den vorhandenen
  Exp.-2b-CSV-Artefakten, ohne neue Power-Flow-Solves, Szenarien,
  AD-Gradienten oder Finite-Difference-Berechnungen zu starten.
- `tests/test_exp02_plot_outputs.py` - leichte Tests fuer Importierbarkeit,
  Label-Mapping, robuste Log-Darstellung, Heatmap-Aggregation,
  Boxplot-Gruppierung, FD-Step-Plot und Figure-Export.

### Erzeugte Artefakte
- `experiments/results/exp02_example_simple_gradients/figures/fig01_ad_vs_fd_parity_by_observable.png/pdf`
- `experiments/results/exp02_example_simple_gradients/figures/fig01a_ad_vs_fd_parity_global.png/pdf`
- `experiments/results/exp02_example_simple_gradients/figures/fig02_gradient_error_heatmap.png/pdf`
- `experiments/results/exp02_example_simple_gradients/figures/fig03_relative_error_boxplot.png/pdf`
- `experiments/results/exp02_example_simple_gradients/figures/fig04_fd_step_study.png/pdf`
- `experiments/results/exp02_example_simple_gradients/figures/fig05_error_by_scenario.png/pdf`
- `experiments/results/exp02_example_simple_gradients/figures/README.md`

### Aktualisierung Fig. 1
- Die Haupt-Parity-Abbildung ist jetzt ein facettierter 2x2-Plot nach
  `output_observable`. Innerhalb jedes Panels werden Eingangsparameter farblich
  und Szenarien ueber Markerformen unterschieden; `n`, maximaler relativer
  Fehler und Medianfehler werden je Observable annotiert.
- Der fruehere globale Parity-Plot bleibt als optionale kompakte Zusatzansicht
  `fig01a_ad_vs_fd_parity_global.png/pdf` erhalten.

### Datenquellen und Grenzen
- Gelesen werden ausschliesslich `gradient_table.csv`, `error_summary.csv` und
  `fd_step_study.csv` unter
  `experiments/results/exp02_example_simple_gradients/`.
- Die Plot-Pipeline bleibt reine Auswertung vorhandener Artefakte und haengt
  damit von deren Spaltenstruktur und numerischer Qualitaet ab.
- Der numerische JAX-Kern, Solver, Residuen, Observables, Gradient-Check-Logik
  und der pandapower-Adapter bleiben unveraendert.

## 2026-05-01 - Experiment 3 Visualisierungen aus bestehenden Artefakten

### Neue Dateien
- `experiments/plot_exp03_figures.py` - erzeugt drei wissenschaftliche
  Matplotlib-Abbildungen direkt aus den vorhandenen Exp.-3-CSV-Artefakten,
  ohne neue Power-Flow-Solves oder neue Szenarien zu starten.
- `tests/test_exp03_plot_outputs.py` - leichte Import-, Filter-, Pivot- und
  Dateierzeugungstests fuer die Plot-Pipeline.

### Erzeugte Artefakte
- `experiments/results/exp03_cross_domain_pv_weather/figures/fig01_t_amb_sweep_p_slack.png/pdf`
- `experiments/results/exp03_cross_domain_pv_weather/figures/fig02_heatmap_g_t_p_slack_base.png/pdf`
- `experiments/results/exp03_cross_domain_pv_weather/figures/fig03_sensitivity_p_slack_vs_t_amb.png/pdf`
- `experiments/results/exp03_cross_domain_pv_weather/figures/README.md`

### Auswertungsumfang
- Abbildung 1 nutzt `scenario_grid.csv` mit
  `weather_case_type == "sweep_1d"` und `observable == "p_slack_mw"`.
- Abbildung 2 nutzt `scenario_grid.csv` mit
  `weather_case_type == "grid_2d"`, `network_scenario == "base"` und
  `observable == "p_slack_mw"`.
- Abbildung 3 nutzt `sensitivity_table.csv` mit
  `weather_case_type == "sweep_1d"`, `observable == "p_slack_mw"` und
  `input_parameter == "t_amb_c"`.
- Die Visualisierung bleibt reine Auswertung bestehender Artefakte; keine
  Aenderungen am PV-Modell oder numerischen Kern.

## 2026-05-01 - Experiment 3: Cross-Domain PV Weather Sensitivity

### Neue Dateien
- `experiments/exp03_cross_domain_pv_weather.py` - Pflicht-Experiment fuer
  Wetter-Sensitivitaetsanalyse auf `example_simple()` mit dem V2-PV-Wettermodell.
  Untersucht die Kette `g_poa_wm2, t_amb_c, wind_ms -> T_cell -> P_pv, Q_pv ->
  NetworkParams -> AC-Power-Flow -> elektrische Observables`. Deckt 3
  Betriebspunkte x (25 2D-Gitter + 6 1D-Sweep) = 93 Forward-Solves ab, berechnet
  1116 AD-Sensitivitaetszeilen und enthaelt einen gezielten AD-vs-FD-Spot-Check
  fuer 4 repraesentative Wettergradienten.
- `tests/test_exp03_cross_domain_outputs.py` - Schema- und Smoke-Tests fuer
  Experiment 3: Importierbarkeit, Spaltendefinitionen, Export-Pipeline und
  Pflichtartefakte ohne schweren Solver-Vollauf.
- `docs/context/experiment_03_plan.md` - Detaillierter Experimentplan nur fuer
  Exp. 3 mit Methodik, Szenariodesign, Artefakten, Grenzen und Zielgroessen.

### Geaenderte Dateien
- `docs/context/experiment_plan.md` - Exp.-3-Abschnitt aktualisiert: neue
  Wetterkette mit `g_poa_wm2`, `t_amb_c`, `wind_ms` (statt G + direktem T_cell),
  1D-Pflichtsweep ueber `t_amb_c`, feste Behandlung von `alpha` und `kappa`,
  Visualisierungen nur als geplante spaetere Auswertung.

### Modellannahmen Exp. 3
- PV-Anlage als wetterabhaengige PQ-Einspeisung am Bus `"MV Bus 2"` (kein PV-Bus).
- `alpha = 1.0` und `kappa = -0.25` als feste Konstanten; keine Variation in Exp. 3.
- Wettergroessen in fachlichen Einheiten (W/m², °C, m/s) ausserhalb des p.u.-Systems.
- Scope-matched: aktiver `gen` wird zu `sgen(P, Q=0)` konvertiert.
- Keine echte PV-Bus-Spannungsregelung, keine Q-Limits, keine PV-PQ-Umschaltung.
- `wind_adj = wind_ms` in der NOCT-SAM-Formel (keine Hoehen-/Montagekorrektur).

### Artefakte (werden durch Ausfuehren des Skripts erzeugt)
- `experiments/results/exp03_cross_domain_pv_weather/scenario_grid.csv/json`
- `experiments/results/exp03_cross_domain_pv_weather/sensitivity_table.csv/json`
- `experiments/results/exp03_cross_domain_pv_weather/gradient_spotcheck.csv/json`
- `experiments/results/exp03_cross_domain_pv_weather/run_summary.csv/json`
- `experiments/results/exp03_cross_domain_pv_weather/metadata.json`
- `experiments/results/exp03_cross_domain_pv_weather/README.md`

## 2026-05-01 - PV-Upstream-Modell V2 mit NOCT-SAM-Zelltemperatur

### Geaenderte Dateien
- `src/diffpf/models/pv.py` - JAX-kompatible
  `cell_temperature_noct_sam(...)`-Funktion und
  `pv_pq_injection_from_weather(...)` als additive Wetter-API ergaenzt
- `src/diffpf/models/__init__.py` - neue NOCT-SAM-Konstanten und Funktionen
  exportiert
- `tests/test_pv_model.py` - Tests fuer Zelltemperatur-Referenzpunkt,
  Monotonien gegenueber Einstrahlung, Umgebungstemperatur und Wind,
  Autodiff sowie wetterbasierte P/Q-Injektion ergaenzt

### Modellannahmen
- Die bestehende V1-API mit explizitem `T_cell` bleibt erhalten.
- Wettergroessen bleiben in fachlichen Einheiten
  (`g_poa_wm2`, `t_amb_c`, `wind_ms`) und werden nicht in das p.u.-System des
  Netzkerns verschoben.
- In der ersten NOCT-SAM-Variante gilt bewusst `wind_adj = wind_ms`; es gibt
  keine Hoehen-, Montage- oder Anlagenkorrektur.
- Die elektrische Kopplung bleibt unveraendert: PQ-Einspeisung am Bus
  `"MV Bus 2"` mit `Q_pv = kappa * P_pv`, keine PV-Bus-Regelung, keine
  Q-Limits, keine PV-PQ-Umschaltung und keine Controllerlogik.
- Es wird weiterhin keine harte oder glatte Saettigungslogik angewendet.

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
