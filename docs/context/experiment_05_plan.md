# Experiment 5 Plan: Screening und PV-Curtailment-Optimierung

## Ziel

Experiment 5 demonstriert, wie eine vorgelagerte PV-Variable gradientenbasiert
so angepasst werden kann, dass eine elektrische Zielgroesse im Netz erreicht
wird. Es ist kein vollstaendiger OPF, sondern ein bewusst kleines
End-to-End-Demonstrationsproblem:

```text
Wetter + Curtailment-Faktor
    -> PV-P/Q-Einspeisung
    -> AC-Power-Flow
    -> Slack-Export / Spannung / Verluste
    -> Gradient nach Curtailment
```

Der wissenschaftliche Kern ist der Nachweis, dass der unveraenderte
differenzierbare Power-Flow-Kern fuer eine gekoppelte Optimierungsfrage genutzt
werden kann.

## Demonstrator und Modellscope

Experiment 5 verwendet ausschliesslich `pandapower.networks.example_simple()`
im bestehenden scope-matched Modell.

Festgelegt sind:

- Kopplungsbus: `"MV Bus 2"`,
- ersetztes Element: `sgen "static generator"`,
- Referenzwerte des ersetzten `sgen`: `P = 2.0 MW`, `Q = -0.5 MVAr`,
- Blindleistungsverhaeltnis der PV-Einspeisung: `Q/P = -0.25`,
- PV als wetterabhaengige P/Q-Einspeisung, nicht als spannungsregelnder PV-Bus.

Der numerische Core bleibt unveraendert. Insbesondere werden keine Aenderungen
an `core/`, `solver/`, `compile/`, Y-Bus, Residuenformulierung oder
Newton-/Implicit-Solver vorgenommen.

## Experiment 5a: Screening und Fallauswahl

Exp. 5a ist ein reines Forward-Screening. Es fuehrt keine Optimierung durch,
sondern identifiziert kritische Betriebspunkte fuer die spaetere
Curtailment-Optimierung.

### Szenariogitter

Das Screening nutzt:

- `load_multiplier_mv_bus_2`: `0.40`, `0.70`, `1.00`, `1.30`,
- `g_poa_wm2`: `200.0`, `600.0`, `1200.0`,
- `t_amb_c`: `-10.0`, `5.0`, `25.0`, `45.0`,
- `wind_ms`: konstant `2.0`,
- `curtailment_factor`: konstant `1.0`,
- `pv_size_factor`: konstant `1.0`,
- `kappa`: konstant `-0.25`.

Damit entstehen `4 x 3 x 4 = 48` PV-Screeningfaelle. Zusaetzlich gibt es je
Lastniveau einen no-PV-Referenzfall. Diese Referenzen dienen zur Berechnung von
Deltas, zum Beispiel fuer Slack-Leistung, Export, Spannung am PV-Bus und
Wirkverluste.

### Bewertung

Exp. 5a bewertet die Betriebspunkte ueber demonstratorinterne Stressindikatoren:

- numerische Konvergenz und Residualnorm,
- Export am Slack beziehungsweise Netzanschlusspunkt,
- Spannung am Kopplungsbus und maximale Netzspannung,
- Verlustdelta gegen no-PV,
- Trafo-Scheinleistungsdelta als Proxy,
- Flussbetraege auf aktiven Leitungen als diagnostische Proxies.

Diese Indikatoren sind keine normativen Netzcode- oder
Betriebsmittelgrenzverletzungen.

### Top-20-Sensitivitaeten

Aus den 48 Screeningfaellen werden die Top-20 nach Kritikalitaet selektiert.
Nur fuer diese Top-20 werden lokale AD-Sensitivitaeten gegen den
`curtailment_factor` berechnet. Dadurch bleibt Exp. 5a ein Screening-Experiment
und vermeidet eine unnoetig breite Sensitivitaetsstudie.

### Separater realistischer Auswahlfall

Zusaetzlich zum unveraenderten 48-Fall-Screening wird ein separater
realistischerer Sommer-Hoch-PV-Fall berechnet:

```text
case_id = selected_realistic_load0p4_g1200_t30
load_multiplier_mv_bus_2 = 0.4
g_poa_wm2 = 1200.0
t_amb_c = 30.0
wind_ms = 2.0
curtailment_factor = 1.0
pv_size_factor = 1.0
kappa = -0.25
```

Dieser Fall ist nicht Teil des 48-Fall-Screeninggitters. Er dient als
fachlich plausiblerer Hauptfall fuer Exp. 5b. Die sehr kalten
`G = 1200 W/m2`, `T_amb = -10 degC`-Faelle bleiben im Screening als
mathematische Stresspunkte erhalten, sind aber nicht das Hauptnarrativ der
Optimierung.

## Experiment 5b: PV-Curtailment-Optimierung

Exp. 5b optimiert genau den ausgewaehlten 30-C-Fall aus Exp. 5a. Es gibt keine
Optimierung ueber mehrere Wetter- oder Lastszenarien.

### Optimierungsvariable

Die physikalische Variable ist der PV-Curtailment-Faktor `c in [0, 1]`:

```text
c = 1.0  -> volle verfuegbare PV-Leistung
c = 0.0  -> vollstaendige Abregelung
P_pv(c) = c * P_pv_available
Q_pv(c) = -0.25 * P_pv(c)
```

Optimiert wird jedoch ein freier Skalar `theta`:

```text
c(theta) = sigmoid(theta)
```

Dadurch bleibt `c` waehrend des gesamten Optimierungsverlaufs im zulaessigen
Intervall.

### Exportziel

Die Zielgroesse ist der Export am Slack beziehungsweise Netzanschlusspunkt.
Die Vorzeichenkonvention lautet:

```text
p_slack_mw < 0  -> Export ins vorgelagerte Netz
p_export_mw = max(0, -p_slack_mw)
```

Fuer den differenzierbaren Optimierungspfad wird der glatte Proxy
`export_proxy_mw = -p_slack_mw` verwendet. Das ist fuer den ausgewaehlten Fall
geeignet, weil dieser exportdominiert ist.

Der demonstratorinterne Zielwert lautet:

```text
p_export_limit_mw = 7.0
```

Dieser Zielwert ist kein normativer Netzcode-Grenzwert.

### Zielfunktion

Die Zielfunktion kombiniert eine glatte Exportverletzungsstrafe mit einer
Regularisierung, die unnoetige Abregelung vermeidet:

```text
violation_mw = softplus(beta * (export_proxy_mw - p_export_limit_mw)) / beta
objective = (violation_mw / p_scale_mw)^2 + lambda_curtailment * (1 - c)^2
```

Der aktuelle Lauf nutzt:

```text
beta = 300
p_scale_mw = 1.0
lambda_curtailment = 1e-4
```

Die Zielfunktion trackt `p_export_target_mw = 6.99 MW`, also einen Zielwert
knapp unterhalb der harten `7.0 MW`-Grenze. Zusaetzlich wird mit `beta = 300`
eine schaerfere glatte Softplus-Penalty gegen die Exportgrenze genutzt. Die
Artefakte exportieren sowohl `hard_export_violation_mw` als auch
`soft_export_violation_mw`, damit der Unterschied zwischen Berichtsgrenze und
glatter AD-Penalty transparent bleibt.

### Optimierer und Referenz

Exp. 5b verwendet einen kleinen lokal implementierten Adam-Loop ohne neue
Dependencies. Zusaetzlich wird eine eindimensionale Grid-Referenz ueber
`curtailment_factor in [0, 1]` berechnet. Die Grid-Referenz dient nur als
Plausibilitaetscheck fuer dieses eindimensionale Demonstrationsproblem, nicht
als skalierbarer Optimierer.

## Experiment 5c: NN-PV-Curtailment-Optimierung

Exp. 5c loest dieselbe Optimierungsaufgabe wie Exp. 5b fuer denselben
ausgewaehlten Fall `selected_realistic_load0p4_g1200_t30`, ersetzt aber den
vorgelagerten analytischen PV-Wetterblock durch das trainierte NN-PV-Surrogat
aus Experiment 4.

Unveraendert gegenueber Exp. 5b bleiben:

- `p_export_limit_mw = 7.0`,
- `p_export_target_mw = 6.99`,
- `beta = 300`,
- `lambda_curtailment = 1e-4`,
- Sigmoid/Logit-Parametrisierung von `c in [0, 1]`,
- lokal implementierter Adam-Loop,
- 1001-Punkte-Grid-Referenz,
- Definition von `p_export_mw`, `hard_export_violation_mw` und
  `soft_export_violation_mw`.

Die NN-Kopplung lautet:

```text
g_poa_wm2, t_amb_c, wind_ms
    -> Exp.-4-NN-Surrogat
    -> P_NN_mw
P_pv_mw(c)    = c * P_NN_mw
Q_pv_mvar(c) = -0.25 * P_pv_mw(c)
```

Da Experiment 4 aktuell keinen standalone Parametercheckpoint persistiert,
reproduziert Exp. 5c den deterministischen Exp.-4-Trainingslauf im Prozess und
verwendet die von `train_surrogate(..., return_diagnostics=True)`
zurueckgegebenen besten globalen Validation-Parameter. Es werden keine
zufaelligen oder untrainierten NN-Parameter verwendet.

Der elektrische Kern bleibt unveraendert; Exp. 5c demonstriert nur, dass die
gekoppelte Optimierung auch mit einem differenzierbaren NN-Upstream-Modell
funktioniert.

## Artefakte

Exp. 5a schreibt nach:

```text
experiments/results/exp05a_network_screening/
```

Wichtige Artefakte:

- `screening_results.csv/json`,
- `top_critical_cases.csv/json`,
- `sensitivity_top20.csv/json`,
- `selected_realistic_case.csv/json`,
- `selected_realistic_case_sensitivity.csv/json`,
- `branch_flows.csv/json`,
- `run_summary.csv/json`,
- `metadata.json`,
- `README.md`.

Exp. 5b schreibt nach:

```text
experiments/results/exp05b_optimize_pv_curtailment/
```

Wichtige Artefakte:

- `selected_case_baseline.csv/json`,
- `optimization_trace.csv/json`,
- `final_solution.csv/json`,
- `grid_reference.csv/json`,
- `constraint_diagnostics.csv/json`,
- `run_summary.csv/json`,
- `metadata.json`,
- `README.md`.

Exp. 5c schreibt nach:

```text
experiments/results/exp05c_optimize_pv_curtailment_nn/
```

Wichtige Artefakte:

- `selected_case_baseline.csv/json`,
- `optimization_trace.csv/json`,
- `final_solution.csv/json`,
- `grid_reference.csv/json`,
- `constraint_diagnostics.csv/json`,
- `run_summary.csv/json`,
- `metadata.json`,
- `README.md`.

Exp.-5c-Figuren werden separat geschrieben nach:

```text
experiments/results/exp05c_figures/
```

Artefakte:

- `fig51_screening_export_overview.png/pdf`,
- `fig53_export_before_after_reference.png/pdf`,
- `fig54_grid_reference_export_vs_curtailment.png/pdf`,
- `fig55_optimization_trace_export_and_curtailment.png/pdf`,
- `README.md`,
- `figure_metadata.json`.

## Aktueller Ergebnisstand

Der ausgewaehlte 30-C-Fall verletzt bei voller PV den Zielwert:

```text
p_export_mw = 7.599971
p_slack_mw  = -7.599971
p_pv_mw     = 2.146286
q_pv_mvar   = -0.536571
```

Bei `c = 0.0` ist die Grenze erreichbar:

```text
p_export_mw = 5.462384
```

Die finale Exp.-5b-Loesung erreicht:

```text
curtailment_factor = 0.714203
PV-Nutzung         = 71.4203 %
PV-Abregelung      = 28.5797 %
p_export_mw        = 6.990006
Export-Margin      = 0.009994 MW
Hard Violation     = 0.000000 MW
Soft Violation     = 0.000162 MW
Grid-Referenz c    = 0.718000
```

Die finale Exp.-5c-Loesung mit NN-Upstream erreicht:

```text
curtailment_factor = 0.719208
PV-Nutzung         = 71.9208 %
PV-Abregelung      = 28.0792 %
p_export_mw        = 6.990006
Export-Margin      = 0.009994 MW
Hard Violation     = 0.000000 MW
Soft Violation     = 0.000162 MW
Grid-Referenz c    = 0.723000
```

Die Werte unterscheiden sich leicht von Exp. 5b, weil das NN-Surrogat das
analytische PV-Modell approximiert. Die Grid-Referenz wurde fuer Exp. 5c neu
berechnet und nicht aus Exp. 5b kopiert.

## Grenzen

Nicht enthalten sind:

- vollstaendige PV-Bus-Spannungsregelung,
- Q-Limits,
- PV-PQ-Umschaltung,
- Controllerlogik,
- Optimierung ueber mehrere Last- oder Wetterszenarien,
- normative thermische Betriebsmittelgrenzwertbewertung.

Alle kritischen Zustaende in Exp. 5a und der Exportzielwert in Exp. 5b/5c sind
demonstratorinterne Indikatoren. Das NN in Exp. 5c ist ein synthetisches
Distillation-Surrogat, kein Messdaten-Prognosemodell; es ist P-only und koppelt
Q deterministisch ueber `Q = -0.25 * P`.
