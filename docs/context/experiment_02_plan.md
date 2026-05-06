# Experiment 2 - Detaillierter Plan: example_simple() Gradientenvalidierung

## Wissenschaftliches Ziel

Experiment 2 validiert die lokalen impliziten Gradienten des differenzierbaren
AC-Power-Flow-Kerns auf `pandapower.networks.example_simple()`. Geprueft wird,
ob Automatic Differentiation durch den impliziten Newton-Solver dieselben
Sensitivitaeten liefert wie zentrale finite Differenzen.

Der Plan beschreibt nur die Experimentdurchfuehrung auf `example_simple()`.
Der historische 3-Bus-PoC bleibt ausserhalb dieses Detailplans.

## Demonstrator und Modellscope

- **Netz:** `pandapower.networks.example_simple()`.
- **Modus:** `scope_matched`.
- **Generatorbehandlung:** Der aktive `gen` wird wie in Experiment 1 zu
  `sgen(P=gen.p_mw, Q=0)` konvertiert.
- **Topologie:** konstant fuer alle AD- und FD-Punkte.
- **Initialisierung:** trafo-shift-aware Initialisierung aus Experiment 1.

Damit wird nicht die vollstaendige `pandapower`-Generatorsemantik validiert,
sondern der im Projekt verwendete PQ-Scope, der auch die Basis fuer Experiment
3 und Experiment 4 bildet.

## Elektrische Betriebspunkte

| Szenario | Lastfaktor | sgen-Faktor | Zweck |
|----------|------------|-------------|-------|
| `base` | 1.00 | 1.00 | Nominalzustand |
| `load_high` | 1.25 | 1.00 | Erhoehte Last |
| `sgen_high` | 1.00 | 1.50 | Erhoehte statische Einspeisung |

Die Szenarien sind bewusst kompakt gewaehlt. Sie decken den Nominalpunkt sowie
zwei einfache Betriebsvariationen ab, ohne den FD-Aufwand durch einen grossen
Szenarioraum unnoetig zu erhoehen.

## Eingangsparameter

Alle untersuchten Eingangsparameter sind kontinuierliche dimensionslose
Skalierungsfaktoren mit `theta0 = 1.0`.

| Parameter | Einheit | Bedeutung |
|-----------|---------|-----------|
| `load_scale_mv_bus_2` | dimensionslos | Skaliert P und Q der vorhandenen Last an `"MV Bus 2"` |
| `sgen_scale_static_generator` | dimensionslos | Skaliert P und Q des vorhandenen `sgen "static generator"` |
| `shunt_q_scale` | dimensionslos | Skaliert die Suszeptanz des vorhandenen Shunts |
| `trafo_x_scale` | dimensionslos | Skaliert die Serienreaktanz des vorhandenen Zweiwicklungs-Transformators |

Die Parameter werden direkt im `NetworkParams`-Pytree variiert. Dadurch bleibt
die Gradientenberechnung JAX-kompatibel und die Topologie unveraendert.

## Ausgangsobservables

| Observable | Einheit | Beschreibung |
|------------|---------|--------------|
| `vm_mv_bus_2_pu` | p.u. | Spannungsbetrag am Bus `"MV Bus 2"` nach Bus-Fusion |
| `p_slack_mw` | MW | Slack-Wirkleistung |
| `total_p_loss_mw` | MW | Gesamtwirkleistungsverluste aus der geloesten Busbilanz |
| `p_trafo_hv_mw` | MW | Wirkleistung, die auf der HV-Seite des Trafos aufgenommen wird |

Diese vier Observables decken lokale Spannung, Systembilanz, Verluste und den
zentralen Trafozweig des `example_simple()`-Netzes ab.

## Pflicht-Gradienten

Fuer jedes Szenario, jeden Eingangsparameter und jedes Observable wird ein
lokaler Gradient berechnet:

```text
d_observable / d_input_parameter
```

Der Pflichtumfang ist:

```text
3 Szenarien x 4 Eingangsparameter x 4 Observables = 48 Gradienten
```

Die AD-Seite nutzt `jax.grad(...)` ueber
`solve_power_flow_implicit(...)`. Der Solvergradient entsteht damit ueber die
implizite Differenzierung am Konvergenzpunkt und nicht durch Unrolling der
Newton-Iterationen.

## Finite-Difference-Vergleich

Der Hauptvergleich verwendet zentrale finite Differenzen mit:

| Groesse | Wert |
|---------|------|
| `theta0` | `1.0` |
| Default-Schrittweite | `1e-4` |
| Formel | `(f(theta0 + h) - f(theta0 - h)) / (2h)` |

Fuer jede Zeile werden AD-Gradient, FD-Gradient, absoluter Fehler, relativer
Fehler, Konvergenzflags, Residualnormen und Iterationszahlen exportiert.

## FD-Schrittweitenstudie

Zusaetzlich wird eine kleine Schrittweitenstudie fuer drei repraesentative
Gradienten durchgefuehrt:

| Auswahl | Szenario | Eingangsparameter | Observable |
|---------|----------|-------------------|------------|
| 1 | `base` | `load_scale_mv_bus_2` | `vm_mv_bus_2_pu` |
| 2 | `base` | `sgen_scale_static_generator` | `p_slack_mw` |
| 3 | `base` | `shunt_q_scale` | `total_p_loss_mw` |

Verwendete Schrittweiten:

```text
1e-2, 1e-3, 1e-4, 1e-5, 1e-6
```

Die Studie dient dazu, das typische FD-Verhalten sichtbar zu machen:
mittlere Schrittweiten liefern stabile Vergleiche, waehrend zu kleine
Schrittweiten Rundungsfehler verstaerken koennen.

## Solver-Einstellungen

| Option | Wert |
|--------|------|
| `max_iters` | 50 |
| `tolerance` | `1e-10` |
| `damping` | `0.7` |
| Initialisierung | `trafo_shift_aware` |
| JAX-Dtype | `float64` |

Die trafo-shift-aware Initialisierung wird aus Experiment 1 uebernommen, weil
`example_simple()` einen 150-Grad-Phasenschieber enthaelt und Flat Start dort
divergieren kann.

## Erwartete Ergebnisbewertung

Alle 48 Pflichtgradienten sollen gueltig sein:

- kein AD-Ausfall,
- kein FD-Ausfall,
- konvergente Basis-, Plus- und Minus-Solves,
- Residualnormen in der Groessenordnung von `4e-11`,
- absolute AD-vs-FD-Fehler im Bereich von etwa `1e-9`.

Die gelieferten Artefakte zeigen als Maximalwert einen absoluten Fehler von
ungefaehr `4.7e-9`. Relative Fehler koennen bei Shunt-Sensitivitaeten groesser
wirken, weil einige Gradienten sehr nahe bei null liegen; entscheidend ist
dann der gleichzeitig sehr kleine absolute Fehler.

## Artefakte

Ordner: `experiments/results/exp02_example_simple_gradients/`

| Datei | Format | Beschreibung |
|-------|--------|--------------|
| `gradient_table.csv/json` | tidy | Eine Zeile pro Szenario, Eingangsparameter und Observable; AD-vs-FD-Vergleich |
| `error_summary.csv/json` | tidy | Aggregierte Fehlerkennzahlen pro Szenario und Eingangsparameter |
| `fd_step_study.csv/json` | tidy | Schrittweitenstudie fuer drei repraesentative Gradienten |
| `metadata.json` | JSON | Reproduzierbarkeitsdaten, Solveroptionen, Parameterlisten und ausgeschlossene Gradienten |
| `README.md` | Text | Menschenlesbare Beschreibung der Ergebnisdateien |

### Tidy-Format-Regeln

- Eine Zeile = ein Gradient oder ein Step-Study-Punkt.
- `scenario`, `input_parameter`, `output_observable`, `ad_grad`, `fd_grad`,
  `abs_error`, `rel_error` und Konvergenzflags sind explizite Spalten.
- Keine verschachtelten Listen in CSV-Zellen.

## Bewusste Ausschluesse und Grenzen

| Ausschluss | Begruendung |
|------------|------------|
| Vollstaendige Jacobian-Matrix aller Netzparameter | Zu breit fuer den PoC; Fokus auf repraesentativen fachlichen Parametern |
| Einzelne Leitungsparameter | Nicht Teil des Pflichtumfangs |
| Einzelne Bus-P/Q-Eintraege | Fachliche Skalierungsparameter sind besser interpretierbar |
| Trafo-Phasenverschiebung und Tap-Ratio | Methodisch sensibel; fuer diesen Validierungslauf nicht variiert |
| Generator-Q-Limits und PV-Bus-Setpoints | Ausserhalb des aktuellen `diffpf`-Scopes |
| Controllerlogik und PV-PQ-Umschaltung | Nicht implementiert |
| `trafo_r_scale` | Als diagnostischer Parameter bewusst nicht Teil des Pflichtumfangs |

## Abgrenzung zu Experiment 1

Experiment 1 validiert den Forward-Solve gegen `pandapower`. Experiment 2
setzt diesen validierten `scope_matched`-Netzzustand voraus und validiert die
lokale Differenzierbarkeit des Solvers.

Die FD-Werte in Experiment 2 sind keine neue `pandapower`-Referenz. Sie sind
ein numerischer Vergleich innerhalb desselben `diffpf`-Modells, bei dem nur
die Berechnungsmethode des lokalen Gradienten wechselt.

## Abgrenzung zu Experiment 3

Experiment 2 variiert elektrische Modellparameter direkt im Netz-Pytree.
Experiment 3 fuegt den vorgelagerten meteorologischen Eingangsraum hinzu und
untersucht die Kette Wetter -> PV-Modell -> P/Q-Injektion -> Power Flow.

Experiment 2 ist damit die numerische Grundlage fuer den spaeteren
Cross-Domain-Sensitivitaetsnachweis, aber noch kein Upstream-Experiment.

## Umgesetzte Visualisierung der Artefakte

Die Auswertung der bereits berechneten Exp.-2-Artefakte erfolgt ueber
`experiments/plot_exp02_gradient_figures.py`. Das Skript liest ausschliesslich
`gradient_table.csv`, `error_summary.csv` und `fd_step_study.csv`; es startet
keine neuen Power-Flow-Solves, AD-Gradienten oder Finite-Difference-Laeufe.

Erzeugte Abbildungen im Ordner
`experiments/results/exp02_example_simple_gradients/figures/`:

- `fig01_ad_vs_fd_parity_by_observable.png/pdf`: facettierter AD-vs-FD-
  Paritaetsplot nach Observable.
- `fig01a_ad_vs_fd_parity_global.png/pdf`: optionale globale Paritaetsansicht.
- `fig02_gradient_error_heatmap.png/pdf`: Heatmap der relativen Fehler je
  Eingangs- und Ausgangsgroesse.
- `fig03_relative_error_boxplot.png/pdf`: Verteilung relativer Fehler nach
  Observable.
- `fig04_fd_step_study.png/pdf`: relative Fehler ueber die FD-Schrittweiten.
- `fig05_error_by_scenario.png/pdf`: Fehlerzusammenfassung nach Szenario.
- `README.md`: Beschreibung der Figuren und Datenquellen.
