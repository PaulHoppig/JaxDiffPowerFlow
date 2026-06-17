# Experiment 5 - Screening und PV-Curtailment-Optimierung

## 1. Zielsetzung

Experiment 5 zeigt, wie der differenzierbare Power-Flow-Kern fuer eine
gekoppelte Optimierungsfrage genutzt werden kann. Die vorgelagerte Variable ist
ein PV-Curtailment-Faktor `c in [0, 1]`. Dieser Faktor skaliert die
wetterabhaengige PV-Einspeisung, die anschliessend in den AC-Power-Flow
eingespeist wird. Die Optimierung soll den Export am Slack beziehungsweise am
Netzanschlusspunkt auf einen demonstratorinternen Zielbereich bringen.

Das Experiment ist kein vollstaendiger OPF. Es optimiert genau eine
Upstream-Variable fuer einen ausgewaehlten Betriebspunkt und verzichtet auf
normative Betriebsmittelgrenzwerte, Controllerlogik, Q-Limits und PV-PQ-
Umschaltung.

Experiment 5 besteht aus vier zusammenhaengenden Teilen:

| Teil | Zweck |
|---|---|
| 5a | Forward-Screening und Auswahl eines geeigneten Hoch-PV-Falls. |
| 5b | Gradientbasierte Curtailment-Optimierung mit analytischem PV-Modell. |
| 5c | Dieselbe Optimierung mit NN-Upstream-Surrogat aus Experiment 4. |
| 5d | Einfache quadratische Zielwertsuche auf 7.0 MW mit analytischem PV-Modell. |

## 2. Metriken und Evaluierungskriterien

Fuer das Screening werden demonstratorinterne Stressindikatoren ausgewertet:

| Metrik | Bedeutung |
|---|---|
| `p_export_mw` | Exportproxy `max(0, -p_slack_mw)`. |
| `vm_mv_bus_2_pu`, `max_vm_pu` | Spannung am Kopplungsbus und maximale Netzspannung. |
| `total_p_loss_mw` | Gesamtwirkverluste. |
| `s_trafo_hv_mva` | Trafo-Scheinleistungsproxy auf der HV-Seite. |
| `criticality_score` | Interner Ranking-Score; kein Netzcode-Kriterium. |

Fuer die Optimierungen sind zentral:

| Metrik | Bedeutung |
|---|---|
| `final_curtailment_factor` | Optimierter PV-Nutzungsfaktor. |
| `final_p_export_mw` | Export nach Optimierung. |
| `final_export_margin_mw` | Abstand zur 7.0-MW-Grenze; positiv bedeutet unterhalb der Grenze. |
| `final_hard_export_violation_mw` | Reporting-Groesse `max(0, p_export_mw - 7.0)`. |
| `final_soft_export_violation_mw` | Glatte Softplus-Verletzung in 5b/5c; in 5d nicht Teil der Zielfunktion. |
| `grid_best_curtailment_factor` | 1001-Punkte-Grid-Referenz fuer die eindimensionale Aufgabe. |
| `abs_c_difference_optimizer_vs_grid` | Abstand zwischen Optimierergebnis und Grid-Referenz. |

## 3. Versuchsaufbau und Durchfuehrung

Alle Teile verwenden `example_simple()` im `scope_matched`-Modus. Der
Kopplungsbus ist `"MV Bus 2"`, das ersetzte Element ist
`sgen "static generator"`, und das feste Blindleistungsverhaeltnis ist
`Q/P = -0.25`.

### Experiment 5a

Das Screening umfasst 48 PV-Faelle:

| Parameter | Werte |
|---|---|
| `load_multiplier_mv_bus_2` | `0.4`, `0.7`, `1.0`, `1.3` |
| `g_poa_wm2` | `200.0`, `600.0`, `1200.0` |
| `t_amb_c` | `-10.0`, `5.0`, `25.0`, `45.0` |
| `wind_ms` | `2.0` |
| `curtailment_factor` | `1.0` |
| `pv_size_factor` | `1.0` |
| `kappa` | `-0.25` |

Zusaetzlich werden vier No-PV-Referenzfaelle berechnet, also insgesamt 52
Forward-Faelle. Fuer die Top-20 nach `criticality_score` werden lokale AD-
Sensitivitaeten nach `curtailment_factor` berechnet. Separat wird ein
realistischeres Sommer-Hoch-PV-Szenario definiert:

```text
selected_realistic_load0p4_g1200_t30
load_multiplier_mv_bus_2 = 0.4
g_poa_wm2 = 1200.0
t_amb_c = 30.0
wind_ms = 2.0
curtailment_factor = 1.0
```

Dieser Fall ist nicht Teil des 48er-Gitters und wird als Hauptfall fuer 5b bis
5d genutzt.

### Experiment 5b und 5c

Die Optimierungsvariable ist ein freier Skalar `theta`, aus dem
`c = sigmoid(theta)` gebildet wird. Dadurch bleibt der Curtailment-Faktor im
Intervall `[0, 1]`. Beide Varianten starten bei `c_init = 0.8`, nutzen einen
lokal implementierten Adam-Loop mit `learning_rate = 0.05` und `max_iter = 300`
sowie eine 1001-Punkte-Grid-Referenz.

Die Zielfunktion in 5b und 5c kombiniert Zieltracking auf `6.99 MW`, eine
glatte Softplus-Strafe gegen die 7.0-MW-Grenze und eine kleine Regularisierung
gegen unnoetige Abregelung:

```text
objective =
  ((export_proxy_mw - 6.99) / 1.0)^2
  + (softplus(300 * (export_proxy_mw - 7.0)) / 300 / 1.0)^2
  + 1e-4 * (1 - c)^2
```

5b nutzt das analytische PV-Wettermodell. 5c nutzt das trainierte
`nn_p_only_fixed_kappa`-Surrogat aus Experiment 4. Da Experiment 4 keinen
standalone Parametercheckpoint persistiert, reproduziert 5c den
deterministischen Exp.-4-Trainingslauf im Prozess und nutzt den besten
Validation-Checkpoint.

### Experiment 5d

5d nutzt wieder das analytische PV-Modell, ersetzt aber die Zielfunktion durch
eine einfache quadratische Zielwertsuche:

```text
objective = ((p_export_proxy_mw - 7.0) / 1.0) ** 2
```

Es gibt keine Softplus-Penalty, kein 6.99-MW-Ziel und keine
Curtailment-Regularisierung. Deshalb ist 5d als symmetrische Zielwertsuche zu
verstehen, nicht als harte einseitige Ungleichungsoptimierung.

## 4. Ergebnisse

### Screening 5a

Alle 52 Forward-Faelle konvergieren. Das Screening enthaelt 48 PV-Faelle, 4
No-PV-Referenzen, 20 Top-Faelle fuer AD-Sensitivitaeten und 160
Sensitivitaetszeilen.

Die Maximalwerte im Screening sind:

| Groesse | Wert | Fall |
|---|---:|---|
| `criticality_score` | `38.25361613630297` | `screen_load0p4_g1200_tneg10` |
| `p_export_mw` | `7.981432223275318` MW | `screen_load0p4_g1200_tneg10` |
| `s_trafo_hv_mva` | `8.169119400658053` MVA | `screen_load0p4_g1200_tneg10` |
| `total_p_loss_mw` | `0.07364034158196842` MW | `screen_load1p3_g1200_tneg10` |
| `vm_mv_bus_2_pu` | `1.0164428537490715` p.u. | `ref_load0p4_no_pv` |
| `max_vm_pu` | `1.0210313555532493` p.u. | `ref_load0p4_no_pv` |

Der ausgewaehlte realistische Fall bei `30 degC` konvergiert in 24 Iterationen
mit Residualnorm `4.304791152458836e-11`. Bei voller PV erzeugt er
`p_pv_mw = 2.146285714285714`, `q_pv_mvar = -0.5365714285714285` und
`p_export_mw = 7.599971141724146`. Damit liegt er oberhalb der
demonstratorinternen 7.0-MW-Grenze und eignet sich fuer die Curtailment-
Optimierung.

Die lokale Sensitivitaet des ausgewaehlten Falls bei `c = 1.0` zeigt:
`d p_export_mw / d c = 2.132930907737494` MW,
`d vm_mv_bus_2_pu / d c = -0.0013420550635065101` p.u.,
`d total_p_loss_mw / d c = 0.013354806548222268` MW und
`d s_trafo_hv_mva / d c = 2.233451265972661` MVA.

### Optimierung 5b

Mit analytischem PV-Modell betraegt der Export bei voller PV
`7.599971141724146` MW und bei `c = 0` `5.462384282296652` MW. Der Fall ist
damit durch Abregelung loesbar.

Das Optimierergebnis ist:

| Groesse | Wert |
|---|---:|
| `final_curtailment_factor` | `0.7142034114138576` |
| PV-Nutzung | `71.42034114138576` % |
| PV-Abregelung | `28.579658858614245` % |
| `final_p_export_mw` | `6.990005680657161` MW |
| `final_export_margin_mw` | `0.00999431934283912` MW |
| `final_hard_export_violation_mw` | `0.0` MW |
| `final_soft_export_violation_mw` | `0.0001622274674944078` MW |
| `objective_reduction_pct` | `99.98710840342818` % |
| Grid-Referenz `c` | `0.718` |
| Abstand zum Grid | `0.003796588586142402` |

Die Grid-Referenz bei `c = 0.718` erreicht `p_export_mw =
6.998113595125772` MW und bleibt damit ebenfalls unter 7.0 MW.

### Optimierung 5c

Mit NN-Upstream betraegt der Full-PV-Export `7.579483241033376` MW und der
Zero-PV-Export `5.476903803760313` MW. Die Differenz zu 5b folgt aus der
Approximation des analytischen PV-Modells durch das NN.

Das finale Ergebnis ist:

| Groesse | Wert |
|---|---:|
| `final_curtailment_factor` | `0.7192083240661113` |
| PV-Nutzung | `71.92083240661114` % |
| PV-Abregelung | `28.079167593388867` % |
| `final_p_export_mw` | `6.990005686429702` MW |
| `final_export_margin_mw` | `0.009994313570297564` MW |
| `final_hard_export_violation_mw` | `0.0` MW |
| `final_soft_export_violation_mw` | `0.00016222774170722822` MW |
| `objective_reduction_pct` | `99.98543092538623` % |
| Grid-Referenz `c` | `0.723` |
| Abstand zum Grid | `0.0037916759338886274` |

Das NN-Surrogat fuehrt damit zu einem sehr aehnlichen Optimierungsverhalten
wie das analytische Modell, aber mit leicht hoeherem optimalem
Curtailment-Faktor.

### Zielwertsuche 5d

In 5d liegt der Full-PV-Export bei `7.6144414198533354` MW und der Zero-PV-
Export bei `5.476903803760313` MW. Die einfache quadratische Zielfunktion
fuehrt zu:

| Groesse | Wert |
|---|---:|
| `final_curtailment_factor` | `0.7121007257421481` |
| PV-Nutzung | `71.2100725742148` % |
| PV-Abregelung | `28.789927425785187` % |
| `final_p_export_mw` | `7.000000018580143` MW |
| `final_objective` | `3.4522170489594787e-16` |
| `final_grad_theta` | `1.6269494605434805e-08` |
| `final_hard_export_violation_mw` | `1.8580142757684825e-08` MW |
| Grid-Best `c` | `0.712` |
| Grid-Best Export | `6.999784912051948` MW |
| Abstand zum Grid | `0.00010072574214814445` |

Das `final_solution.csv` markiert `constraint_satisfied = False`, weil der
Reporting-Wert um `1.8580142757684825e-08` MW oberhalb von 7.0 MW liegt. Die
`constraint_diagnostics.csv` ordnet den Fall gleichzeitig als erwartbar
loesbar ein. Der Grund ist die Zielfunktion: 5d minimiert symmetrisch den
Abstand zu 7.0 MW und enthaelt keine harte Ungleichungsstrafe. Die finale
Ueberschreitung ist numerisch winzig.

## 5. Diskussion und Interpretation

Experiment 5 zeigt den praktischen Nutzen der zuvor validierten
Differenzierbarkeit. Das Screening identifiziert exportlastige Hoch-PV-Faelle,
und die Optimierungsvarianten koennen den ausgewaehlten Fall gezielt in die
Nahe der 7.0-MW-Marke bringen.

5b und 5c sind besonders vergleichbar, weil Zielfunktion, Optimierer und
Grid-Referenz gleich bleiben. Beide erreichen rund `6.990006` MW Export,
keine harte Grenzverletzung und eine Zielreduzierung von etwa 99.99 %. Dass
5c mit dem NN-Upstream eine leicht andere PV-Nutzung findet, ist kein Fehler,
sondern die erwartete Folge des trainierten Surrogats.

5d ist methodisch anders zu lesen. Die einfache quadratische Zielfunktion
zielt direkt auf 7.0 MW und landet deshalb naeher an der Grenze als 5b/5c.
Sie bietet eine klare Plausibilitaetskontrolle fuer die eindimensionale
Optimierung, ersetzt aber keine einseitige Grenzoptimierung.

Die Grenzen des Experiments bleiben bewusst eng: Es gibt nur einen
Optimierungsfall, keine Mehrszenario-Optimierung, keine thermischen
Betriebsmittelgrenzwerte, keine Spannungsregelung, keine Q-Limits und keine
Controllerlogik. Der 7.0-MW-Wert ist ein demonstratorinterner Zielwert, kein
normativer Netzanschlussgrenzwert.

Verwendete Artefakte: `experiments/results/exp05a_network_screening/`,
`experiments/results/exp05b_optimize_pv_curtailment/`,
`experiments/results/exp05c_optimize_pv_curtailment_nn/`,
`experiments/results/exp05d_optimize_pv_curtailment_simple_objective/`.
