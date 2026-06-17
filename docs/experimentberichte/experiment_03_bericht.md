# Experiment 3 - Cross-Domain-Sensitivitaet Wetter zu Netz

## 1. Zielsetzung

Experiment 3 demonstriert die End-to-End-Differenzierbarkeit von
meteorologischen Eingangsgroessen ueber ein PV-Modell bis in den stationaeren
AC-Power-Flow. Im Gegensatz zu Experiment 2 werden nicht mehr nur elektrische
Netzparameter variiert. Stattdessen wird die Kette

```text
g_poa_wm2, t_amb_c, wind_ms
  -> Zelltemperatur
  -> PV-P/Q-Einspeisung
  -> NetworkParams
  -> AC-Power-Flow
  -> elektrische Observables
```

ausgewertet. Ziel ist der Nachweis, dass `diffpf` lokale Sensitivitaeten
elektrischer Groessen gegenueber Wettergroessen direkt per AD berechnen kann.

## 2. Metriken und Evaluierungskriterien

Bewertet werden drei Ebenen:

| Ebene | Artefakt | Kriterium |
|---|---|---|
| Forward-Solve | `scenario_grid.csv` | Konvergenz und plausible Wertebereiche der elektrischen Observables. |
| Sensitivitaeten | `sensitivity_table.csv` | Vollstaendige AD-Sensitivitaeten fuer alle Wetterfaelle, Observables und Wetterinputs. |
| Spot-Check | `gradient_spotcheck.csv` | AD-vs-FD-Vergleich fuer vier repraesentative Wettergradienten. |

Die betrachteten Observables sind `vm_mv_bus_2_pu`, `p_slack_mw`,
`total_p_loss_mw` und `p_trafo_hv_mw`. Die Wetterinputs sind `g_poa_wm2`,
`t_amb_c` und `wind_ms`.

## 3. Versuchsaufbau und Durchfuehrung

Das Netz ist erneut `example_simple()` im `scope_matched`-Modus. Der
Kopplungsbus ist `"MV Bus 2"`, und das ersetzte Element ist das
`sgen "static generator"`. Die PV-Anlage wird als wetterabhaengige PQ-
Einspeisung modelliert, nicht als spannungsregelnder PV-Bus. Das feste
Blindleistungsverhaeltnis ist `kappa = -0.25`; `alpha = 1.0` bleibt in diesem
Experiment ebenfalls konstant.

Das PV-Modell besteht aus einer reduzierten NOCT-SAM-Zelltemperaturformel und
einer analytischen PV-Leistungsformel. Die Wettergroessen bleiben in ihren
natuerlichen Einheiten; nur die daraus entstehende P/Q-Einspeisung wird in das
Netzmodell uebertragen.

Die aktuellen Artefakte verwenden drei elektrische Lastszenarien:

| Szenario | Lastfaktor |
|---|---:|
| `base` | 1.00 |
| `load_low` | 0.75 |
| `load_high` | 1.25 |

Pro Lastszenario werden 36 Wetterfaelle berechnet:

| Wetterdesign | Umfang |
|---|---:|
| 2D-Gitter `g_poa_wm2 x t_amb_c` bei `wind_ms = 2.0` | 25 Faelle |
| 1D-Temperatursweep bei `g_poa_wm2 = 800`, `wind_ms = 2.0` | 6 Faelle |
| 1D-Einstrahlungssweep bei `t_amb_c = 25`, `wind_ms = 2.0` | 5 Faelle |

Damit entstehen `3 x 36 = 108` Forward-Solves. Fuer jeden Wetterfall werden
vier Observables exportiert, also 432 Zeilen in `scenario_grid.csv`. Fuer jede
Kombination aus Wetterfall, Observable und Wetterinput entstehen
`108 x 4 x 3 = 1296` Sensitivitaetszeilen.

Der AD-vs-FD-Spot-Check nutzt vier repraesentative Gradienten. Die
FD-Schrittweiten betragen `5.0 W/m^2` fuer Einstrahlung, `0.1 degC` fuer
Temperatur und `0.1 m/s` fuer Wind.

## 4. Ergebnisse

Alle 108 Wetter-Netz-Faelle konvergieren. Die `run_summary.csv` berichtet fuer
jedes der drei Lastszenarien `n_converged = 36` und `n_failed = 0`.

Die Wertebereiche der wichtigsten Forward-Observables ueber alle Faelle sind:

| Observable | Minimum | Maximum |
|---|---:|---:|
| `p_slack_mw` | `-7.065349873857671` MW | `-4.812104162045644` MW |
| `vm_mv_bus_2_pu` | `1.0028270477050107` p.u. | `1.0111085789426586` p.u. |
| `total_p_loss_mw` | `0.045253532920366` MW | `0.0558571815775237` MW |
| `p_trafo_hv_mw` | `-7.060535592034766` MW | `-4.805946577640785` MW |

Die PV-Wirkleistung reicht in den Lastszenario-Zusammenfassungen von
`0.3622857142857143` MW bis `2.0171428571428573` MW. Die Spannung am
Kopplungsbus ist im `load_low`-Szenario am hoechsten
(`1.0101310033554507` bis `1.0111085789426586` p.u.) und im
`load_high`-Szenario am niedrigsten (`1.0028270477050107` bis
`1.0037991761183949` p.u.).

Die lokalen Sensitivitaeten von `p_slack_mw` zeigen plausible Richtungen:

| Sensitivitaet | Wertebereich |
|---|---:|
| `d p_slack_mw / d g_poa_wm2` | `-0.0020989628971381` bis `-0.0015249819192953` MW pro W/m^2 |
| `d p_slack_mw / d t_amb_c` | `0.0015961665665379` bis `0.0079641645460634` MW pro degC |
| `d p_slack_mw / d wind_ms` | `-0.0406334925819562` bis `-0.0016287413944264` MW pro m/s |

Mehr Einstrahlung und mehr Wind erhoehen die PV-Leistung und machen
`p_slack_mw` negativer, also exportlastiger. Hoehere Umgebungstemperatur
reduziert wegen des Temperaturkoeffizienten die PV-Leistung und verschiebt
`p_slack_mw` in positive Richtung.

Der AD-vs-FD-Spot-Check umfasst 4 Gradienten. Der groesste absolute Fehler
liegt bei `1.2740646282286602e-08`; der groesste relative Fehler bei
`0.0008119990806768`. Beide Werte treten beim Wind-Gradienten von
`vm_mv_bus_2_pu` im Basisszenario auf. Fuer einen Spot-Check der erweiterten
Wetterkette ist das numerisch konsistent mit den Ergebnissen aus Experiment 2.

## 5. Diskussion und Interpretation

Experiment 3 zeigt, dass der differenzierbare Power-Flow-Kern nicht auf rein
elektrische Eingangsparameter beschraenkt ist. Die Wetter-PV-Netz-Kette ist
durchgaengig differenzierbar, liefert konvergente Forward-Solves und erzeugt
physikalisch plausible Sensitivitaetsrichtungen.

Die Interpretation der Slack-Leistung ist zentral: Negative `p_slack_mw`-
Werte bedeuten Export ins vorgelagerte Netz. Hoehere Einstrahlung und bessere
Kuehlung durch Wind erhoehen den Export, waehrend hoehere Temperatur wegen der
geringeren PV-Wirkleistung den Export reduziert. Diese Muster erscheinen in
den AD-Sensitivitaeten konsistent.

Das Experiment bleibt bewusst ein Demonstrator. Die PV-Anlage ist eine PQ-
Einspeisung ohne Spannungsregelung, Q-Limits, Controllerlogik oder PV-PQ-
Umschaltung. Ausserdem ist der Szenarioraum kompakt und auf `example_simple()`
beschraenkt. Die Aussage ist daher kein allgemeines PV-Ertrags- oder
Netzbetriebsmodell, sondern ein Nachweis der differenzierbaren Kopplung.

Verwendete Artefakte: `experiments/results/exp03_cross_domain_pv_weather/`.
