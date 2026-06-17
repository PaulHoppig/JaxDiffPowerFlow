# Numerische Detailwerte und Laufzeitentscheidungen

Stand: 2026-06-15

Dieses Dokument sammelt die konkreten numerischen Detailwerte, Toleranzen,
Konvergenzkriterien, Initialisierungen und JAX/JIT-Entscheidungen des
Softwareprojekts. Es ergaenzt `docs/context/designentscheidungen.md`.

Bei Widerspruechen zwischen Kontextdokumenten gilt der `CHANGELOG.md` als
hoechste textuelle Wahrheit. Fuer tatsaechlich ausgefuehrte numerische Werte
ist der Code massgeblich.

## 1. Globale numerische Grundentscheidung

| Thema | Wert / Entscheidung | Ort | Begruendung |
|---|---:|---|---|
| JAX-Praezision | `jax_enable_x64 = True` | `src/diffpf/__init__.py`, Experimente | Power-Flow-Rechnung, Sensitivitaeten und finite Differenzen sind empfindlich gegen Rundungsfehler. `float64` reduziert Abweichungen gegen pandapower und stabilisiert Gradienten. |
| Reelle State-Vektoren | `float64` | `src/diffpf/core/state.py` | Spannungen werden als Real- und Imaginaerteile gefuehrt, damit JAX-Jacobians und Newton-Loeser direkt auf reellen Vektoren arbeiten. |
| Komplexe Netzadmittanz | `complex128` | `src/diffpf/core/params.py`, `src/diffpf/core/ybus.py` | Admittanzen bleiben physikalisch natuerlich komplex, waehrend der Solver intern reell differenziert. |
| Index-Datentypen | `int32` | `src/diffpf/core/topology.py` | JAX-freundliche kompakte Indizes fuer scatter/add und Maskenoperationen. |

## 2. Newton-Solver

| Parameter | Default | Typische Experimentwerte | Ort |
|---|---:|---:|---|
| Maximale Iterationen | `25` | `50` in Exp01-Exp05, teils `100` in Diagnosevarianten | `src/diffpf/solver/newton.py`, `experiments/*.py` |
| Residualtoleranz | `1e-10` | `1e-10`, teils `1e-12` in strikteren Diagnosen | `src/diffpf/solver/newton.py`, Experimente |
| Daempfung | `1.0` | `0.7` in den meisten Experimenten | `src/diffpf/solver/newton.py`, Experimente |
| Residualnorm | L2-Norm | L2-Norm | `src/diffpf/solver/newton.py` |
| Konvergenzbedingung | `residual_norm <= tolerance` | gleich | `src/diffpf/solver/newton.py` |
| Abbruchschleife | solange `iteration < max_iters` und `residual_norm > tolerance` | gleich | `src/diffpf/solver/newton.py` |
| Newton-Update | `x_next = x - damping * step` | gleich | `src/diffpf/solver/newton.py` |
| Residual-Loss | `0.5 * ||r||^2` | gleich | `src/diffpf/solver/newton.py` |

Die Standardwerte sind streng genug fuer Sensitivitaetsrechnungen und zugleich
noch robust in kleinen Netzen. Die Experimente verwenden meistens `50`
Iterationen und `damping = 0.7`, weil der Beispieltransformator mit grossem
Phasenwinkel und die PV-Szenarien numerisch anspruchsvoller sind als ein
einfacher Standardfall.

## 3. Implizite Differenzierung und lineare Loesung

| Thema | Wert / Entscheidung | Ort | Zweck |
|---|---|---|---|
| Implizite Loesung | `jax.lax.custom_root` | `src/diffpf/solver/implicit.py` | Trennt Forward-Newton von der Rueckwaertsableitung und vermeidet Differenzieren durch alle Newton-Iterationen. |
| Tangentenloesung | Dichte Jacobi-Matrix ueber Einheitsbasis und `jax.vmap` | `src/diffpf/solver/implicit.py` | Fuer kleine Netze einfach, deterministisch und gut validierbar. |
| Linearer Solver | `jnp.linalg.solve(jacobian, cotangent)` | `src/diffpf/solver/implicit.py` | Direkte Loesung statt iterativem Verfahren, passend zur kleinen Problemgroesse. |
| Jacobian im Newton | `jax.jacfwd` | `src/diffpf/solver/newton.py` | Vorwaertsmodus ist fuer die kleinen State-Vektoren uebersichtlich und stabil. |

## 4. JIT-Entscheidungen

| Bereich | Entscheidung | Ort | Begruendung |
|---|---|---|---|
| Kernsolver | JAX-pure und jit-faehig, keine globale Pflicht-JIT | `src/diffpf/solver/*.py` | Tests und Experimente koennen entscheiden, ob Kompilierung lohnt. Das haelt Debugging einfacher. |
| Newton-Schleife | `jax.lax.while_loop` | `src/diffpf/solver/newton.py` | JIT-kompatible Schleifenlogik ohne Python-Control-Flow auf Tracer-Werten. |
| Exp05b Objective-Gradient | `jax.jit(jax.value_and_grad(...))` | `experiments/exp05b_optimize_pv_curtailment.py` | Adam-Optimierung ruft denselben Objective oft auf; JIT amortisiert die Kosten. |
| Exp05b Case-Evaluator | `@jax.jit` | `experiments/exp05b_optimize_pv_curtailment.py` | Schnelle wiederholte Auswertung fuer Verlauf, Grid und Ergebnisdiagnostik. |
| Exp05c Objective-Gradient | JIT analog zu Exp05b | `experiments/exp05c_optimize_pv_curtailment_nn.py` | Gleiche Optimierungsstruktur, aber mit NN-Surrogat als Upstream-Modell. |
| Exp05d Objective-Gradient | `jax.jit(jax.value_and_grad(...))` | `experiments/exp05d_optimize_pv_curtailment_simple_objective.py` | Schnelle Gradienten fuer die vereinfachte Zielfunktion. |
| Exp05d Case-Evaluator | `@jax.jit` | `experiments/exp05d_optimize_pv_curtailment_simple_objective.py` | Konsistente, kompilierte Auswertung fuer Optimierung und Grid-Vergleich. |
| NN-Inferenz | JIT-testbar | `tests/test_pq_surrogate_model.py` | Sicherstellung, dass das Surrogat in JAX-kompilierten Pipelines nutzbar ist. |
| pandapower-Referenz | `numba = False` | `experiments/exp01_validate_example_simple.py` | Reproduzierbare Referenzlaeufe und weniger Abhaengigkeit von optionaler Beschleunigung. |

## 5. Initialisierung und Import-Schwellen

| Thema | Wert | Ort | Bedeutung |
|---|---:|---|---|
| Flat Start Realteil | `vr = 1.0` fuer Nicht-Slack-Busse | `src/diffpf/io/pandapower_import.py` | Standardinitialisierung nahe Nennspannung. |
| Flat Start Imaginaerteil | `vi = 0.0` fuer Nicht-Slack-Busse | `src/diffpf/io/pandapower_import.py` | Nullwinkel als einfacher Default. |
| Trafo-Shift-aware Start | LV-Winkel = Slack-Winkel minus Trafo-Shift | Exp01-Exp05 Hilfslogik | Noetig, weil der Beispieltrafo `150 deg` Phasenschieber enthaelt. |
| Nullimpedanz-Grenze | `abs(z) < 1e-12` | `src/diffpf/core/ybus.py` | Schutz vor Division durch null bzw. nahezu null bei Leitungen und Transformatoren. |
| Generator-Spannungsabgleich | Mismatch-Schwelle `1e-9` | `src/diffpf/io/pandapower_import.py` | Erkennt widerspruechliche Generator-Spannungsvorgaben am selben Bus. |
| Logit-Epsilon | `1e-12` | Exp05b/Exp05d Optimierung | Verhindert unendliche Logits bei Curtailment-Werten exakt `0` oder `1`. |

## 6. pandapower-Referenzwerte

| Einstellung | Wert | Ort |
|---|---:|---|
| Algorithmus | `nr` | `experiments/exp01_validate_example_simple.py` |
| Spannungswinkel | `calculate_voltage_angles = True` | `experiments/exp01_validate_example_simple.py` |
| Initialisierung | `init = "dc"` | `experiments/exp01_validate_example_simple.py` |
| Toleranz | `tolerance_mva = 1e-9` | `experiments/exp01_validate_example_simple.py` |
| Max. Iterationen | `50` | `experiments/exp01_validate_example_simple.py` |
| Trafo-Modell | `trafo_model = "pi"` | `experiments/exp01_validate_example_simple.py` |
| Numba | `False` | `experiments/exp01_validate_example_simple.py` |

Validierungshilfen verwenden typischerweise
`NewtonOptions(max_iters=30, tolerance=1e-10, damping=1.0)`. Die Experimente
verwenden fuer schwierigere Szenarien meist `50 / 1e-10 / 0.7`.

## 7. PV-Modell

| Parameter | Wert | Ort | Bedeutung |
|---|---:|---|---|
| Kopplungsbus | `MV Bus 2` | `src/diffpf/models/pv.py` | Bus fuer die PV-Einspeisung im Beispielnetz. |
| Kopplungs-SGen | `static generator` | `src/diffpf/models/pv.py` | Pandapower-Element, das durch PV-Leistung ersetzt/skaliert wird. |
| Basis-Wirkleistung | `2.0 MW` | `src/diffpf/models/pv.py` | Nenn-PV-Leistung im Beispiel. |
| Basis-Blindleistung | `-0.5 Mvar` | `src/diffpf/models/pv.py` | Entspricht konstantem Blindleistungsverhaeltnis. |
| Blindleistungsverhaeltnis | `q / p = -0.25` | `src/diffpf/models/pv.py` | Konstante Q/P-Kopplung. |
| Referenzirradianz | `1000 W/m^2` | `src/diffpf/models/pv.py` | STC-nahe Normierung. |
| Referenztemperatur | `25 deg C` | `src/diffpf/models/pv.py` | STC-Zelltemperaturreferenz. |
| Temperaturkoeffizient | `-0.004 1/deg C` | `src/diffpf/models/pv.py` | Lineare temperaturbedingte Leistungsreduktion. |
| Irradianzexponent | `alpha = 1.0` | `src/diffpf/models/pv.py` | Lineare Abhaengigkeit von Einstrahlung. |
| NOCT-Anpassung | `45.0 deg C` | `src/diffpf/models/pv.py` | Zelltemperaturmodell. |
| Referenzwirkungsgrad | `0.18` | `src/diffpf/models/pv.py` | Bestandteil der Zelltemperaturabschaetzung. |
| Tau-Alpha | `0.90` | `src/diffpf/models/pv.py` | Optischer Absorptions-/Transmissionsfaktor. |
| Windterm | `9.5 / (5.7 + 3.8 * wind_ms)` | `src/diffpf/models/pv.py` | Windkuehlung im NOCT-SAM-Modell. |
| Clipping | keines | `src/diffpf/models/pv.py` | Bewusst glatte, differenzierbare Kennlinie. |

## 8. Experiment 1: Lastflussvalidierung

| Wert | Einstellung |
|---|---|
| Newton | `max_iters=50`, `tolerance=1e-10`, `damping=0.7` |
| Referenz | pandapower `nr`, `init="dc"`, `tolerance_mva=1e-9`, `max_iteration=50`, `trafo_model="pi"` |
| Szenarien | `base`, `load_low=0.75`, `load_high=1.25`, `sgen_low=0.50`, `sgen_high=1.50`, kombinierte Hoch-/Niedrigfaelle |
| Referenzmodi | `scope_matched`, `original_pandapower` |
| Initialisierung | Trafo-Shift-aware wegen `150 deg` Trafo-Phasenverschiebung |

## 9. Experiment 2: Gradientenvalidierung

| Thema | Wert |
|---|---:|
| Newton | `max_iters=50`, `tolerance=1e-10`, `damping=0.7` |
| Default-FD-Schritt | `1e-3` |
| FD-Schrittstudie | `1e-2`, `1e-3`, `1e-4`, `1e-5`, `1e-6` |
| Detaillierte FD-Schritte | von `1e+0` bis `1e-10` in Zwischenstufen |
| Relative-Error-Floor | `1e-12` |
| Szenarien | `base`, `load_high`, `sgen_high` |
| Inputs | `load_scale_mv_bus_2`, `sgen_scale_static_generator`, `shunt_q_scale`, `trafo_x_scale` |
| Outputs | `vm_mv_bus_2_pu`, `p_slack_mw`, `total_p_loss_mw`, `p_trafo_hv_mw` |

Die vielen FD-Schritte dienen nicht dem Produktivlauf, sondern der Diagnose:
zu grosse Schritte messen Nichtlinearitaet, zu kleine Schritte Rundungsfehler.

## 10. Experiment 3: PV-Wetter-Kopplung

| Thema | Wert |
|---|---:|
| Newton | `max_iters=50`, `tolerance=1e-10`, `damping=0.7` |
| PV-Irradianzexponent | `alpha = 1.0` |
| Q/P-Verhaeltnis | `kappa = -0.25` |
| Irradianzlevel | `200`, `400`, `600`, `800`, `1000 W/m^2` |
| Temperaturlevel | `5`, `15`, `25`, `35`, `45 deg C` |
| Referenzwind | `2.0 m/s` |
| Referenzirradianz fuer Sweeps | `800 W/m^2` |
| Temperatur-Sweep | `5`, `15`, `25`, `35`, `45`, `55 deg C` |
| Irradianz-Sweep | identisch zu den Irradianzleveln |
| FD-Spotcheck `g_poa_wm2` | `5.0` |
| FD-Spotcheck `t_amb_c` | `0.1` |
| FD-Spotcheck `wind_ms` | `0.1` |
| Elektrische Szenarien | `base=1.00`, `load_low=0.75`, `load_high=1.25` |

Aktueller Code-Stand: Durch den zusaetzlichen Irradianz-Sweep ergeben sich
`36` Wetterfaelle pro elektrischem Szenario, also `108` Forward-Loesungen und
`1296` Sensitivitaetszeilen. Aeltere Kontextstellen mit `93` Loesungen und
`1116` Sensitivitaetszeilen sind damit ueberholt.

## 11. Experiment 4: Modulares Upstream-NN-Surrogat

| Thema | Wert |
|---|---:|
| Newton | `max_iters=50`, `tolerance=1e-10`, `damping=0.7` |
| Q/P-Verhaeltnis | `-0.25` |
| Upstream-Modelle | analytisches PV-Modell, P-only-NN mit fixem Kappa, direkter PQ-Baseline |
| Wetterfaelle | `6` benannte Faelle von `200` bis `1200 W/m^2` |
| Elektrische Szenarien | `base=1.0`, `load_low=0.75`, `load_high=1.25` |
| FD `g_poa_wm2` | `1.0` |
| FD `t_amb_c` | `0.05` |
| FD `wind_ms` | `0.01` |
| Relative-Error-Floor | `1e-12` |

### NN-Trainingswerte

| Parameter | Default |
|---|---:|
| Trainingssamples | `32768` |
| Validierungssamples | `8192` |
| Evaluationssamples | `8192` |
| Start-Lernrate | `8e-2` |
| End-Lernrate | `1e-4` |
| Maximale Trainingsschritte | `8000` |
| Warm-Restart aktiviert | `True` |
| Basis-Trainingsschritte | `8000` |
| Finetuning-Schritte | `8000` |
| Restart-Zyklen | `2000`, `2000`, `2000`, `2000` |
| Restart-LR-Max | `2e-2`, `1e-2`, `5e-3`, `2e-3` |
| Restart-LR-Min | `5e-4`, `2e-4`, `1e-4`, `5e-5` |
| Hidden Width Default | `16` |
| Hidden Layers | `2` |
| Aktivierung | `tanh` |
| Seed | `42` |
| Logging-Intervall | `50` |
| Normalisierung Zentrum | `(600.0, 17.5, 5.25)` |
| Normalisierung Skala | `(600.0, 27.5, 4.75)` |

Parameterzaehlung: Das Width-16-Netz hat `353` Parameter, das Width-8-Netz
`113` Parameter. Biases werden mit null initialisiert; Gewichte nutzen
Glorot/Xavier-uniforme Initialisierung.

### Referenzmetriken fuer Width 16

| Metrik | Wert |
|---|---:|
| Val-MSE | `2.565834826e-04` |
| Val-MAE | `0.024622580 MW` |
| Eval-P-MAE | `0.024815085 MW` |
| Eval-P-RMSE | `0.032617826 MW` |
| Max-P-Error | `0.185804774 MW` |

### Referenzmetriken fuer Width 8

| Metrik | Wert |
|---|---:|
| Val-MSE | `2.3189919622e-04` |
| Val-MAE | `0.023338907 MW` |
| Eval-P-MAE | `0.023523218 MW` |
| Eval-P-RMSE | `0.031040266 MW` |
| Max-P-Error | `0.177669209 MW` |

## 12. Experiment 5a: Netzscreening

| Thema | Wert |
|---|---:|
| Newton | `max_iters=50`, `tolerance=1e-10`, `damping=0.7` |
| Lastmultiplikatoren | `0.40`, `0.70`, `1.00`, `1.30` |
| Irradianzlevel | `200`, `600`, `1200 W/m^2` |
| Temperaturlevel | `-10`, `5`, `25`, `45 deg C` |
| Wind | `2.0 m/s` |
| Curtailment | `1.0` |
| PV-Groessenfaktor | `1.0` |
| Q/P-Verhaeltnis | `-0.25` |
| Screeningfaelle | `48` plus No-PV-Referenzen |
| Auswertung | Top-20 Sensitivitaeten |

Aus dem Screening wird ein realistischer Optimierungsfall mit hoher
PV-Einspeisung und niedriger Last ausgewaehlt:
`selected_realistic_load0p4_g1200_t30`.

## 13. Experiment 5b: PV-Curtailment-Optimierung

| Parameter | Wert |
|---|---:|
| Exportlimit | `7.0 MW` |
| Exportziel | `6.99 MW` |
| Curtailment-Minimum | `0.0` |
| Curtailment-Maximum | `1.0` |
| Initiales Curtailment | `0.8` |
| Lernrate | `0.05` |
| Adam-Iterationen | `300` |
| Adam Beta 1 | `0.9` |
| Adam Beta 2 | `0.999` |
| Adam Epsilon | `1e-8` |
| Softplus-Beta | `300.0` |
| Leistungsskala | `1.0 MW` |
| Curtailment-Regularisierung | `1e-4` |
| Grid-Punkte | `1001` |

Die Zielfunktion kombiniert Zielnaehe zu `6.99 MW`, eine glatte
Softplus-Strafe fuer Ueberschreitungen von `7.0 MW` und eine kleine
Regularisierung gegen unnoetiges Abregeln. Als Exportnaeherung wird
`-p_slack_mw` verwendet; fuer Berichte wird `max(0, -p_slack_mw)` genutzt.

## 14. Experiment 5c: NN-basierte Curtailment-Optimierung

Exp05c uebernimmt die wesentlichen Optimierungswerte aus Exp05b:

| Parametergruppe | Wert |
|---|---|
| Exportlimit und Ziel | `7.0 MW`, `6.99 MW` |
| Curtailment-Bereich | `0.0` bis `1.0`, Start `0.8` |
| Adam | Lernrate `0.05`, `300` Iterationen, Betas `0.9` / `0.999`, Epsilon `1e-8` |
| Softplus und Skala | Beta `300.0`, Leistungsskala `1.0 MW` |
| Regularisierung | `1e-4` |
| Grid | `1001` Punkte |

Der wesentliche Unterschied ist nicht die Optimierung selbst, sondern das
Upstream-Modell: Exp05c nutzt das P-only-NN aus Exp04 und setzt
`Q = -0.25 * P`.

## 15. Experiment 5d: Vereinfachte Zielfunktion

| Parameter | Wert |
|---|---:|
| Exportlimit | `7.0 MW` |
| Exportziel | `7.0 MW` |
| Curtailment-Minimum | `0.0` |
| Curtailment-Maximum | `1.0` |
| Initiales Curtailment | `0.8` |
| Lernrate | `0.05` |
| Adam-Iterationen | `300` |
| Adam Beta 1 | `0.9` |
| Adam Beta 2 | `0.999` |
| Adam Epsilon | `1e-8` |
| Leistungsskala | `1.0 MW` |
| Grid-Punkte | `1001` |
| Konsistenzvergleich | `rtol=1e-10`, `atol=1e-10` |

Die Zielfunktion ist bewusst reduziert:

```text
((p_export_proxy_mw - 7.0) / p_scale_mw) ** 2
```

Es gibt keine Softplus-Strafe und keine Curtailment-Regularisierung. Deshalb
ist `soft_export_violation_mw` in den Ergebnisdateien nicht aussagekraeftig
und wird als `NaN` gefuehrt.

## 16. Werte, die bewusst Modellannahmen statt Solver-Toleranzen sind

| Wert | Einordnung |
|---|---|
| `q / p = -0.25` | Fachliche PV-Leistungsannahme, keine numerische Stabilisierung. |
| Keine PV-Clipping-Grenze | Differenzierbarkeitsentscheidung fuer Experimente, keine physikalische Vollmodellierung. |
| `damping = 0.7` in Experimenten | Numerische Robustheitsentscheidung fuer die Szenarien, kein physikalischer Parameter. |
| `numba = False` bei pandapower | Reproduzierbarkeits-/Abhaengigkeitsentscheidung, kein Rechenergebnisparameter. |
| `1001` Grid-Punkte | Referenzdiskretisierung fuer Optimierungsvergleich, kein kontinuierlicher Loeserparameter. |

## 17. Kurzliste der wichtigsten Zahlen

- Solver-Default: `25` Iterationen, `1e-10` Toleranz, `1.0` Daempfung.
- Experiment-Newton: meist `50` Iterationen, `1e-10` Toleranz, `0.7` Daempfung.
- pandapower-Referenz: `nr`, `init="dc"`, `tolerance_mva=1e-9`, `50` Iterationen, `trafo_model="pi"`.
- Globale Praezision: `jax_enable_x64 = True`.
- Nullimpedanzschutz: `1e-12`.
- Generator-Spannungs-Mismatch: `1e-9`.
- PV: `2.0 MW`, `-0.5 Mvar`, `q/p=-0.25`, `1000 W/m^2`, `25 deg C`, `gamma=-0.004`.
- NN: `32768` Trainingssamples, `8192` Validierungssamples, Hidden Width `16`, `2` Hidden Layers, Seed `42`.
- Exp05 Optimierung: Exportlimit `7.0 MW`, Start-Curtailment `0.8`, Lernrate `0.05`, Adam `300` Iterationen, Grid `1001`.
