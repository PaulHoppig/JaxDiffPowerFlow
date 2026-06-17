# Experiment 1 - Solver-Validierung gegen pandapower

## 1. Zielsetzung

Experiment 1 validiert den stationaeren AC-Power-Flow-Kern von `diffpf` gegen
`pandapower` auf `pandapower.networks.example_simple()`. Der Fokus liegt auf
dem Forward-Solve: Fuer mehrere elektrische Betriebspunkte wird geprueft, ob
`diffpf` dieselben Knotenspannungen, Winkel, Slack-Leistungen, Zweigfluesse und
Verlustgroessen berechnet wie eine passend konfigurierte `pandapower`-Referenz.

Die strikte Validierung erfolgt im Modus `scope_matched`. In diesem Modus wird
der aktive `gen` des pandapower-Netzes in ein statisches Einspeiseelement mit
`P = gen.p_mw` und `Q = 0` ueberfuehrt. Damit vergleichen beide Solver dasselbe
PQ-Modell. Der zweite Modus `original_pandapower` bleibt ein Kontextvergleich:
`pandapower` verwendet dort weiterhin den PV-Bus-Generator, waehrend `diffpf`
keine Generator-Spannungsregelung, Q-Limits oder PV-PQ-Umschaltung modelliert.

## 2. Metriken und Evaluierungskriterien

Die Hauptkriterien sind Konvergenz, Residualnorm und absolute Abweichungen
zwischen `diffpf` und `pandapower`.

| Metrik | Einheit | Bedeutung |
|---|---:|---|
| `diffpf_converged`, `pandapower_converged` | bool | Beide Solver muessen fuer den Vergleich konvergieren. |
| `diffpf_residual_norm` | p.u. | Norm der Residuen am diffpf-Loesungspunkt. |
| `max_vm_pu_abs_diff` | p.u. | Maximale Spannungsbetragsabweichung ueber alle Busse. |
| `max_va_degree_abs_diff` | deg | Maximale Winkelabweichung ueber alle Busse. |
| `p_slack_mw_abs_diff`, `q_slack_mvar_abs_diff` | MW/MVAr | Abweichung der Slack-Wirk- und Blindleistung. |
| `max_line_p_mw_abs_diff`, `max_line_q_mvar_abs_diff` | MW/MVAr | Maximale Leitungsflussabweichung. |
| `max_trafo_p_mw_abs_diff`, `max_trafo_q_mvar_abs_diff` | MW/MVAr | Maximale Trafoflussabweichung. |
| `total_p_loss_mw_abs_diff`, `total_q_loss_mvar_abs_diff` | MW/MVAr | Abweichung der Gesamtverluste. |

Bewertet wird primaer der `scope_matched`-Modus. Der `original_pandapower`-
Modus darf groessere Abweichungen zeigen, weil dort unterschiedliche
Generatorsemantiken verglichen werden.

## 3. Versuchsaufbau und Durchfuehrung

Das Experiment verwendet `example_simple()` mit 7 urspruenglichen Bussen. Nach
der Verarbeitung geschlossener Bus-Bus-Schalter bleiben 5 interne Busse uebrig;
die Fusionsgruppen sind `[[1, 2], [3, 4]]`. Von 4 urspruenglichen Leitungen
sind nach Switch-Verarbeitung 3 aktiv. Zusaetzlich enthaelt das validierte Netz
1 aktiven Zweiwicklungstransformator, 1 Shunt, 1 Last und im `scope_matched`-
Modell 2 statische Einspeisungen.

Es werden sieben Betriebspunkte berechnet:

| Szenario | Lastfaktor | sgen-Faktor |
|---|---:|---:|
| `base` | 1.00 | 1.00 |
| `load_low` | 0.75 | 1.00 |
| `load_high` | 1.25 | 1.00 |
| `sgen_low` | 1.00 | 0.50 |
| `sgen_high` | 1.00 | 1.50 |
| `combined_high_load_low_sgen` | 1.25 | 0.50 |
| `combined_low_load_high_sgen` | 0.75 | 1.50 |

`diffpf` nutzt einen Newton-Solver mit `max_iters = 50`, `tolerance = 1e-10`
und `damping = 0.7`. Die Initialisierung ist `trafo_shift_aware`: HV-Busse
starten am Slack-Winkel, LV-Busse bei `slack_angle - trafo_shift_deg`. Diese
Initialisierung ist fuer das Netz wichtig, weil der Transformator eine
Phasenverschiebung von 150 Grad besitzt.

Die pandapower-Referenz verwendet `algorithm = nr`,
`calculate_voltage_angles = True`, `init = dc`, `tolerance_mva = 1e-9`,
`max_iteration = 50`, `trafo_model = pi` und `numba = False`. Im
`scope_matched`-Modus werden Leitungen mit offenen Line-Switches auch in der
pandapower-Referenz ausser Betrieb gesetzt, damit beide Topologien exakt
uebereinstimmen.

## 4. Ergebnisse

Alle sieben `scope_matched`-Szenarien konvergieren in beiden Solvern. `diffpf`
benoetigt in allen ausgewerteten Zeilen 24 Newton-Iterationen. Die groesste
Residualnorm im strikten Vergleich betraegt `4.321395507015984e-11` im Szenario
`load_high`.

Die aktuellen Hauptartefakte zeigen im `scope_matched`-Modus eine nahezu
maschinengenaue Uebereinstimmung:

| Groesse | Maximaler absoluter Fehler | Worst Case |
|---|---:|---|
| Spannungsbetrag | `4.773959005888173e-14` p.u. | `load_low` |
| Spannungswinkel | `2.1884716261411086e-12` deg | `combined_high_load_low_sgen` |
| Leitungs-Wirkfluss | `1.2182610475974798e-10` MW | `combined_low_load_high_sgen` |
| Leitungs-Blindfluss | `2.889666284033865e-10` MVAr | `combined_low_load_high_sgen` |
| Trafo-Wirkfluss | `6.9180217110442754e-12` MW | `combined_high_load_low_sgen` |
| Trafo-Blindfluss | `4.849454171562684e-13` MVAr | `base` |
| Slack-Wirkleistung | `1.21897159033324e-10` MW | `combined_low_load_high_sgen` |
| Slack-Blindleistung | `2.888715933124786e-10` MVAr | `combined_low_load_high_sgen` |
| Gesamtwirkverluste | `1.1067813332488186e-12` MW | `load_high` |
| Gesamtblindverluste | `1.9060308886764687e-12` MVAr | `load_high` |

Der Kontextvergleich `original_pandapower` zeigt deutlich groessere
Abweichungen. Die maximale Spannungsbetragsabweichung erreicht dort
`0.021483391094873516` p.u.; die maximale Slack-Blindleistungsabweichung
erreicht `4.033259256765676` MVAr. Diese Werte sind keine
Validierungsfehler des Kernmodells, sondern folgen aus dem Vergleich eines
PV-Bus-Modells mit einem PQ-Scope-Modell.

Die Diagnoseartefakte bestaetigen die technische Einordnung. Die globale
Y-Bus-Diagnose weist `max_abs_complex_diff = 0.0`, `frobenius_norm_diff = 0.0`
und keine Eintraege mit Differenzen oberhalb `1e-12` aus. Die historische
Transformator-Magnetisierungsdiagnose dokumentiert, dass der frueher sichtbare
aktive Leistungs-Offset durch die aktuelle pi-Stempelung praktisch nicht mehr
die Hauptaussage bestimmt: Der aktuelle mittlere Baseline-Offset der
Slack-Wirkleistung liegt bei `4.601834131407705e-06` MW und damit unter der
korrigierten Schwelle von `0.001` MW.

## 5. Diskussion und Interpretation

Experiment 1 erfuellt seine Kernanforderung: Innerhalb des dokumentierten
PQ-Modellscopes stimmt `diffpf` mit `pandapower` auf dem verwendeten
Demonstrator praktisch exakt ueberein. Die Fehlergroessen im strikten Modus
liegen deutlich unter fachlich relevanten elektrischen Toleranzen und sind
numerisch durch Rundungs- und Solvergrenzen erklaerbar.

Wichtig ist die klare Trennung der Referenzmodi. Der `scope_matched`-Modus ist
der eigentliche Paritaetstest. Der `original_pandapower`-Modus zeigt dagegen,
welche Abweichungen entstehen, wenn pandapower weiterhin einen
spannungsregelnden Generator modelliert. Diese Abweichungen sind erwartbar und
begruenden zugleich die dokumentierte Modellgrenze: `diffpf` validiert hier
keinen vollstaendigen pandapower-Controller- oder PV-Bus-Scope.

Die Ergebnisse bilden die Grundlage fuer die weiteren Experimente. Experiment 2
kann die impliziten Gradienten auf demselben validierten Modell pruefen.
Experiment 3 bis 5 nutzen den gesicherten AC-Power-Flow-Kern dann fuer
vorgelagerte PV-, Wetter-, Surrogat- und Optimierungsketten.

Verwendete Artefakte: `experiments/results/exp01_example_simple_validation/`,
`experiments/results/exp01_residual_voltage_offset_diagnostic/`,
`experiments/results/exp01_transformer_magnetization_ablation/`.
