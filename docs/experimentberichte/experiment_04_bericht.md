# Experiment 4 - Modulare Upstream-Kopplung mit NN-Surrogat

## 1. Zielsetzung

Experiment 4 untersucht, ob der validierte und differenzierbare
Power-Flow-Kern modular mit unterschiedlichen vorgelagerten Modellen gekoppelt
werden kann. Die zentrale Frage ist nicht, ob ein neuronales Netz ein
allgemeines PV-Prognosemodell ersetzt, sondern ob ein trainiertes
JAX-Surrogat dieselbe P/Q-Schnittstelle wie das analytische PV-Wettermodell
nutzen kann, ohne den AC-Power-Flow-Kern zu veraendern.

Verglichen werden drei Upstream-Modelle:

| Modell | Rolle |
|---|---|
| `analytic_pv_weather` | Referenzmodell aus Experiment 3. |
| `nn_p_only_fixed_kappa` | Trainiertes MLP-Surrogat fuer P; Q folgt aus `Q = -0.25 * P`. |
| `direct_pq_scale_baseline` | Einfache differenzierbare Baseline ohne Temperatur- und Windphysik. |

## 2. Metriken und Evaluierungskriterien

Das Experiment bewertet drei Ebenen:

| Ebene | Metriken |
|---|---|
| Surrogatguete | Val-MSE, Val-MAE, Eval-MAE, Eval-RMSE, maximaler P-Fehler. |
| Netzseitige Wirkung | Absolute und floor-relative Fehler der Power-Flow-Observables gegen `analytic_pv_weather`. |
| Sensitivitaetsmuster | Absolute Gradientenfehler, relative Gradientenfehler, Vorzeichenuebereinstimmung, Magnitudenverhaeltnis und Cosine Similarity. |

Die netzseitige Bewertung nutzt die Observables `vm_mv_bus_2_pu`,
`va_mv_bus_2_deg`, `p_slack_mw`, `q_slack_mvar`, `total_p_loss_mw` und
`p_trafo_hv_mw`. Der analytische PV-Wetterblock ist die Referenz; das NN und
die direkte Baseline werden dagegen verglichen.

## 3. Versuchsaufbau und Durchfuehrung

Das Netz ist weiterhin `example_simple()` im `scope_matched`-Modus. Der
Kopplungsbus ist `"MV Bus 2"`, und die Upstream-Modelle schreiben P/Q ueber
denselben Adapter in `NetworkParams`. Der Power-Flow-Kern wird nicht veraendert.

Das NN ist ein kleines MLP mit 3 Eingaben (`g_poa_wm2`, `t_amb_c`, `wind_ms`),
2 versteckten Schichten, Breite 16, `tanh`-Aktivierungen und einer skalaren
normalisierten P-Ausgabe. Es hat 353 Parameter. Q wird nicht frei gelernt,
sondern deterministisch als `Q = -0.25 * P` gesetzt.

Der Trainingsdatensatz ist synthetisch und wird durch Distillation des
analytischen PV-Wettermodells erzeugt. Die aktuellen Artefakte verwenden:

| Split | Samples | Wetterbereich |
|---|---:|---|
| Train | 32768 | `G: 0..1200 W/m^2`, `T: -10..45 degC`, `Wind: 0.5..10 m/s` |
| Val | 8192 | gleicher Bereich |
| Eval | 8192 | gleicher Bereich |

Die Input-Normalisierung verwendet Zentrum `[600.0, 17.5, 5.25]` und Skalen
`[600.0, 27.5, 4.75]`. Trainiert wird in JAX mit Full-Batch-Gradient-Descent.
Der aktuelle Hauptlauf ist zweiphasig: zuerst 8000 Schritte Cosine Decay
`8e-2 -> 1e-4`, danach 8000 Warm-Restart-Finetune-Schritte in vier Zyklen.
Der beste Checkpoint liegt in Phase `warm_restart_finetune`, Zyklus 4, bei
global Step `16000`.

Fuer den Power-Flow-Vergleich werden 6 repraesentative Wetterfaelle und 3
Netzszenarien genutzt; pro Modell ergeben sich 18 Netzloesungen. In allen
Modellen ist die Konvergenzrate `1.0`, es gibt keine fehlgeschlagenen Solves.

## 4. Ergebnisse

Das NN verbessert die Surrogatguete gegenueber dem dokumentierten
Referenzlauf deutlich. Die aktuellen Werte sind:

| Kennzahl | Referenzlauf | Aktueller Lauf | Relative Verbesserung |
|---|---:|---:|---:|
| Val-MSE | `2.565834826e-04` | `1.0796882754565427e-04` | `0.579205853581885` |
| Val-MAE | `0.02462258` MW | `0.01641394628038067` MW | `0.33337829421690707` |
| Eval P-MAE | `0.024815085` MW | `0.016544012227095853` MW | `0.33330825878308085` |
| Eval P-RMSE | `0.032617826` MW | `0.021139486791535485` MW | `0.3519038702476528` |
| Max. P-Fehler | `0.185804774` MW | `0.11933011475462596` MW | `0.35776615322797917` |

Gegenueber dem frueheren Width-8-Modell verbessert der Width-16-Kandidat
ebenfalls alle dokumentierten Kennzahlen. Die Parameterzahl steigt von 113 auf
353; die Val-MSE-Verbesserung liegt bei `0.5344148263315862`, die Eval-P-MAE-
Verbesserung bei `0.2966943456845125`.

Netzseitig bleibt das NN nahe am analytischen Referenzmodell. Fuer
`nn_p_only_fixed_kappa` betragen die wichtigsten maximalen absoluten Fehler:

| Observable | Max. absoluter Fehler |
|---|---:|
| `vm_mv_bus_2_pu` | `1.1323224377690622e-05` p.u. |
| `va_mv_bus_2_deg` | `0.006066060497687431` deg |
| `p_slack_mw` | `0.018270978680423156` MW |
| `q_slack_mvar` | `0.005973552877499344` MVAr |
| `total_p_loss_mw` | `0.000100783626406864` MW |
| `p_trafo_hv_mw` | `0.018281961098182364` MW |

Die direkte Baseline ist deutlich ungenauer. Ihr maximaler Fehler in
`p_slack_mw` betraegt `0.38187284769201923` MW, in `p_trafo_hv_mw`
`0.3821130002949449` MW und in `vm_mv_bus_2_pu`
`0.0002403174577252365` p.u.

Die Sensitivitaetsauswertung zeigt ein gemischteres, aber interpretierbares
Bild. Fuer das NN stimmen die Vorzeichen bei Einstrahlungs- und
Temperaturgradienten in den ausgewerteten Gruppen vollstaendig
(`sign_match_rate = 1.0`). Bei Windgradienten sinkt die Vorzeichenrate in den
zusammengefassten Gruppen auf `0.8333333333333334`; gleichzeitig sind diese
Gradienten teilweise klein und damit empfindlicher gegen Surrogatfehler.

Die AD-vs-FD-Spotchecks der gekoppelten Modelle sind erfolgreich. Die
Power-Flow-Loesungen der drei Modelle weisen maximale Residualnormen um
`4.3e-11` und maximal 24 Newton-Iterationen auf.

## 5. Diskussion und Interpretation

Experiment 4 belegt die modulare Kopplung: Das analytische Modell, das
NN-Surrogat und die direkte Baseline verwenden denselben P/Q-Adapter und
denselben Power-Flow-Kern. Dass alle Netzloesungen konvergieren und das NN
netzseitig nahe am analytischen Modell bleibt, stuetzt die Entwurfsentscheidung
einer klaren Upstream/Power-Flow-Schnittstelle.

Das NN ist als synthetisches Distillation-Surrogat zu interpretieren. Es lernt
nicht aus Messdaten und ist nicht als allgemeines PV-Prognosemodell gemeint.
Die gute Guete zeigt, dass ein kleines differenzierbares Modell ausreichend
ist, um die Kopplungsmechanik und die Weitergabe von Gradienten zu
demonstrieren.

Die Sensitivitaetsmuster sind fuer Einstrahlung und Temperatur robuster als
fuer Wind. Das ist fachlich plausibel, weil Wind in der PV-Kette indirekt ueber
die Zelltemperatur wirkt und lokale Gradienten kleiner oder nichtlinearer
ausfallen koennen. Fuer spaetere Arbeiten waeren Gradient-Matching-Losses,
persistierte NN-Checkpoints und breitere Validierungsraeume sinnvolle
Erweiterungen.

Verwendete Artefakte:
`experiments/results/exp04_modular_upstream_nn_surrogate/`.
