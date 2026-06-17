# Experiment 2 - Validierung impliziter Gradienten

## 1. Zielsetzung

Experiment 2 validiert die lokalen Gradienten des differenzierbaren
Power-Flow-Kerns. Waehrend Experiment 1 den Forward-Solve gegen `pandapower`
absichert, prueft Experiment 2, ob die ueber den impliziten Newton-Solver
berechneten Automatic-Differentiation-Gradienten mit zentralen finiten
Differenzen innerhalb desselben `diffpf`-Modells uebereinstimmen.

Das Experiment beantwortet damit die Frage, ob der geloeste AC-Power-Flow in
der aktuellen Implementierung nicht nur numerisch korrekt, sondern auch lokal
differenzierbar und fuer Sensitivitaetsanalysen verwendbar ist.

## 2. Metriken und Evaluierungskriterien

Die zentrale Vergleichsgroesse ist je Zeile:

```text
abs_error = |ad_grad - fd_grad|
rel_error = abs_error / max(|fd_grad|, floor)
```

Neben den Fehlern werden Konvergenzflags und Residualnormen fuer Basis-,
Plus- und Minus-Loesung der finiten Differenz ausgewertet. Ein Gradient gilt
als belastbar, wenn AD und FD ohne Ausfall berechnet werden und die beteiligten
Power-Flow-Solves konvergieren.

Die Aggregation erfolgt nach Szenario und Eingangsparameter. Die Artefakte
berichten je Gruppe `n_gradients`, `n_valid`, maximale und mediane absolute
Fehler, maximale und mediane relative Fehler sowie die Anzahl von AD- und
FD-Ausfaellen.

## 3. Versuchsaufbau und Durchfuehrung

Das Experiment verwendet wie Experiment 1 `pandapower.networks.example_simple()`
im `scope_matched`-Modus. Der aktive Generator wird als statische PQ-
Einspeisung modelliert; Topologie, Busfusion und Trafo-Initialisierung bleiben
unveraendert.

Untersucht werden drei Betriebspunkte:

| Szenario | Bedeutung |
|---|---|
| `base` | Nominaler Betriebspunkt |
| `load_high` | Erhoehte Last |
| `sgen_high` | Erhoehte statische Einspeisung |

Die vier Eingangsparameter sind kontinuierliche Skalierungsfaktoren:

| Eingangsparameter | Bedeutung |
|---|---|
| `load_scale_mv_bus_2` | Skaliert P und Q der Last am Kopplungsbus. |
| `sgen_scale_static_generator` | Skaliert P und Q der statischen Einspeisung. |
| `shunt_q_scale` | Skaliert die Shunt-Suszeptanz. |
| `trafo_x_scale` | Skaliert die Trafo-Serienreaktanz. |

Die vier Observables sind `vm_mv_bus_2_pu`, `p_slack_mw`,
`total_p_loss_mw` und `p_trafo_hv_mw`. Daraus entstehen
`3 x 4 x 4 = 48` Pflichtgradienten. Der AD-Pfad nutzt implizite
Differentiation am konvergierten Newton-Loesungspunkt. Die FD-Referenz nutzt
zentrale Differenzen; zusaetzlich wird eine Schrittweitenstudie fuer
repraesentative Gradienten ausgewertet.

## 4. Ergebnisse

Alle 48 Pflichtgradienten wurden ohne AD- oder FD-Ausfall berechnet. In den
aggregierten Fehlerdateien ist fuer jede Szenario-Parameter-Gruppe
`n_valid = 4`, `n_failed_ad = 0` und `n_failed_fd = 0` dokumentiert.

Der groesste absolute AD-vs-FD-Fehler ueber alle Pflichtgradienten betraegt
`1.0911085368547901e-09`. Er tritt im Szenario `load_high` fuer
`load_scale_mv_bus_2 -> p_slack_mw` auf. Der groesste relative Fehler betraegt
`4.5052905981458266e-05` und liegt bei
`base: shunt_q_scale -> p_trafo_hv_mw`. Diese relative Auffaelligkeit ist durch
kleine Gradientenbetraege erklaerbar; der absolute Fehler derselben
Fehlergruppe bleibt mit maximal `2.3550275946482464e-10` sehr klein.

Ausgewaehlte aggregierte Maximalfehler:

| Szenario | Eingangsparameter | Max. absoluter Fehler | Max. relativer Fehler |
|---|---|---:|---:|
| `base` | `load_scale_mv_bus_2` | `5.069467484686285e-10` | `6.233772965735229e-08` |
| `base` | `shunt_q_scale` | `2.3550275946482464e-10` | `4.5052905981458266e-05` |
| `load_high` | `load_scale_mv_bus_2` | `1.0911085368547901e-09` | `6.643076797348457e-08` |
| `load_high` | `trafo_x_scale` | `4.817520288941113e-10` | `4.30299956758662e-07` |
| `sgen_high` | `sgen_scale_static_generator` | `5.288965411465174e-10` | `2.374947562090738e-08` |
| `sgen_high` | `shunt_q_scale` | `3.7722842654886206e-10` | `1.3880869907667885e-05` |

Die groesste Basis-Residualnorm in der Gradiententabelle liegt bei
`4.321395507015984e-11`, identisch mit dem `load_high`-Forward-Solve aus
Experiment 1. Damit entstehen die Gradienten an numerisch sauber geloesten
Arbeitspunkten.

Die detaillierte FD-Schrittweitenstudie enthaelt 31 Schrittweiten pro
ausgewaehltem Gradienten. Fuer
`base: load_scale_mv_bus_2 -> vm_mv_bus_2_pu` liegt die beste Schrittweite bei
`5e-05` mit relativem Fehler `1.3767616424499308e-11`. Fuer
`base: sgen_scale_static_generator -> p_slack_mw` ist `0.001` am besten; die
kleinsten Schrittweiten zeigen dort erwartete Rundungsinstabilitaet. Fuer
`base: shunt_q_scale -> total_p_loss_mw` ist `0.01` am besten; bei extrem
kleiner Schrittweite `2e-09` wird der relative Fehler mit
`1.8949403539381944` deutlich schlechter.

## 5. Diskussion und Interpretation

Experiment 2 bestaetigt die lokale Differenzierbarkeit des validierten
`diffpf`-Power-Flow-Kerns. Die AD-Gradienten stimmen ueber alle 48
Pflichtvergleiche mit zentralen finiten Differenzen ueberein; die absoluten
Fehler liegen im Bereich von etwa `1e-10` bis `1e-09`.

Die relativen Fehler muessen differenziert interpretiert werden. Bei
Shunt-Sensitivitaeten treten groessere relative Werte auf, weil die
Referenzgradienten teilweise nahe bei null liegen. Fuer die numerische
Bewertung ist dort der gleichzeitig sehr kleine absolute Fehler entscheidend.

Die Schrittweitenstudie zeigt das erwartete Verhalten finiter Differenzen:
mittlere Schrittweiten liefern stabile Referenzen, waehrend sehr kleine
Schrittweiten durch Rundung und Subtraktionsausloeschung unguenstig werden
koennen. Damit stuetzt die Studie nicht nur die AD-Ergebnisse, sondern auch die
Wahl einer sinnvollen FD-Referenz.

Das Experiment liefert die methodische Basis fuer die folgenden
Cross-Domain-Experimente. Wenn elektrische Modellparameter korrekt
differenzierbar sind, kann die Kette in Experiment 3 um Wetter- und PV-Modelle
erweitert und in Experiment 5 fuer Optimierung genutzt werden.

Verwendete Artefakte: `experiments/results/exp02_example_simple_gradients/`.
