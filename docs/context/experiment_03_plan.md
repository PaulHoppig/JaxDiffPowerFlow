# Experiment 3 – Detaillierter Plan: Cross-Domain PV Weather Sensitivity

## Wissenschaftliches Ziel

Experiment 3 demonstriert den zentralen wissenschaftlichen Mehrwert von `diffpf`:
die End-to-End-Differenzierbarkeit vom meteorologischen Eingangsraum durch ein
vorgelagertes PV-Modell in einen stationären AC-Power-Flow, sodass Sensitivitäten
elektrischer Observables gegenüber nicht-elektrischen Wettergrößen direkt per
Automatic Differentiation (AD) berechnet werden können.

Konkret wird die Kette

```text
g_poa_wm2, t_amb_c, wind_ms
    -> cell_temperature_noct_sam(...)
    -> pv_pq_injection_from_weather(...)
    -> P_pv, Q_pv am Bus "MV Bus 2"
    -> NetworkParams
    -> AC-Power-Flow (impliziter Newton-Solver)
    -> elektrische Observables
```

auf Plausibilität geprüft und Sensitivitäten werden durch implizite
Differentiation berechnet. Ein kleiner AD-vs-FD-Spot-Check plausibilisiert die
Gradientenkette.

## Demonstrator und Kopplungspunkt

- **Netz:** `pandapower.networks.example_simple()` (scope-matched: aktiver `gen`
  wird zu `sgen(P, Q=0)` konvertiert, konsistent mit Exp. 1 und Exp. 2).
- **Kopplungspunkt:** Bus `"MV Bus 2"`.
- **Ersetztes Element:** `sgen "static generator"` (P = 2.0 MW, Q = -0.5 MVAr,
  Q/P = -0.25).
- Das ersetzte Element wird in der Netzcompilierung deaktiviert. Die PV-Einspeisung
  wird über `inject_pv_at_bus(...)` direkt in `NetworkParams` eingetragen.

## Verwendetes PV-Modell

### V1 – Analytische elektrische Kopplung

```text
P_pv = alpha * P_ref * (G / G_ref) * (1 + gamma_p * (T_cell - T_ref))
Q_pv = kappa * P_pv
```

Referenzwerte: `G_ref = 1000 W/m²`, `T_ref = 25 °C`, `P_ref = 2.0 MW`,
`gamma_p = -0.004 /°C`.

### V2 – NOCT-SAM-Zelltemperaturmodell

`cell_temperature_noct_sam(g_poa_wm2, t_amb_c, wind_ms)` berechnet die
Zelltemperatur nach einer reduzierten NOCT-SAM-Formel:

```text
T_cell = T_amb + (G_poa / 800) * (T_noct_adj - 20)
         * (1 - eta_ref / tau_alpha) * 9.5 / (5.7 + 3.8 * wind_adj)
```

mit Standardparametern `T_noct_adj = 45 °C`, `eta_ref = 0.18`,
`tau_alpha = 0.90`. In dieser Version gilt `wind_adj = wind_ms` (keine
Höhen- oder Montagekorrektur).

### Kopplung

`pv_pq_injection_from_weather(g_poa_wm2, t_amb_c, wind_ms)` kombiniert beide
Modelle: zunächst wird `T_cell` via NOCT-SAM berechnet, dann wird das V1-Modell
mit `T_cell` und `G_poa` ausgewertet. Die Kette ist vollständig JAX-differenzierbar.

## Feste Modellkonstanten in Experiment 3

`alpha` und `kappa` werden in Experiment 3 **nicht** variiert. Sie sind feste,
dokumentierte Modellkonstanten:

| Konstante | Wert | Bedeutung |
|-----------|------|-----------|
| `alpha` | `1.0` | Dimensionsloser Skalierungsfaktor |
| `kappa` | `-0.25` | Q/P-Verhältnis |

Sensitivitäten in Exp. 3 beziehen sich ausschließlich auf Wettergrößen.
`alpha` und `kappa` werden erst in späteren Experimenten (Exp. 4/5) variiert.

## Elektrische Betriebspunkte

| Name | Lastfaktor | Beschreibung |
|------|-----------|--------------|
| `base` | 1.00 | Nominalzustand |
| `load_low` | 0.75 | 25 % reduzierte Last |
| `load_high` | 1.25 | 25 % erhöhte Last |

Die Lastscalierung erfolgt über das `scaling`-Feld der Lastzeile im
pandapower-Netz. Die Topologie bleibt statisch.

## Wetterdesign

### A) 2D-Gitter (Pflicht)

| Parameter | Werte |
|-----------|-------|
| `g_poa_wm2` | 200, 400, 600, 800, 1000 W/m² |
| `t_amb_c` | 5, 15, 25, 35, 45 °C |
| `wind_ms` | 2.0 m/s (fest) |

Anzahl Wetterfälle: 5 × 5 = **25 Punkte** pro Betriebspunkt.

### B) 1D-Temperatursweep (Pflicht)

| Parameter | Werte |
|-----------|-------|
| `g_poa_wm2` | 800 W/m² (fest) |
| `t_amb_c` | 5, 15, 25, 35, 45, 55 °C |
| `wind_ms` | 2.0 m/s (fest) |

Anzahl Wetterfälle: **6 Punkte** pro Betriebspunkt.

**Gesamtzahl Forward-Solves:** 3 Betriebspunkte × (25 + 6) = **93 Solves**.

Der 1D-Sweep über `t_amb_c` ist der Pflicht-Sweep, da er den Einfluss der
Umgebungstemperatur auf Zelltemperatur, PV-Leistung und Netzgrößen am
deutlichsten zeigt.

## Pflicht-Observables

| Observable | Einheit | Beschreibung |
|-----------|---------|--------------|
| `vm_mv_bus_2_pu` | p.u. | Spannungsbetrag am Kopplungsbus MV Bus 2 |
| `p_slack_mw` | MW | Wirkleistung am Slack-Bus (Bilanzquelle) |
| `total_p_loss_mw` | MW | Gesamte Wirkleistungsverluste im Netz |
| `p_trafo_hv_mw` | MW | Wirkleistung an der HV-Seite des Transformators |

Optional sind `q_slack_mvar`, Residualnorm und Konvergenzstatus, die ebenfalls
exportiert werden.

## Pflicht-Sensitivitäten

Für jede Kombination aus Observable und Wettereingang wird ein lokaler Gradient
via reverse-mode AD (implizite Differentiation durch den Newton-Solver) berechnet:

```text
d_vm_mv_bus_2_pu / d_g_poa_wm2
d_vm_mv_bus_2_pu / d_t_amb_c
d_vm_mv_bus_2_pu / d_wind_ms
d_p_slack_mw / d_g_poa_wm2
d_p_slack_mw / d_t_amb_c
d_p_slack_mw / d_wind_ms
d_total_p_loss_mw / d_g_poa_wm2
d_total_p_loss_mw / d_t_amb_c
d_total_p_loss_mw / d_wind_ms
d_p_trafo_hv_mw / d_g_poa_wm2
d_p_trafo_hv_mw / d_t_amb_c
d_p_trafo_hv_mw / d_wind_ms
```

Jeder Backward-Pass durch `jax.grad(..., argnums=(0, 1, 2))` liefert alle drei
Wettergradienten eines Observables in einem Durchlauf.

**Gesamtzahl Sensitivity-Zeilen:** 93 × 4 Observables × 3 Eingänge = **1116 Zeilen**
in `sensitivity_table.csv`.

## Methodische Abgrenzung (AD-vs-FD Spot-Check)

Im Gegensatz zu Experiment 2, das eine systematische AD-vs-FD-Validierung über
48 Gradienten mit Schrittweitenstudie durchführt, beschränkt sich Experiment 3
auf einen **gezielten Spot-Check** von 4 repräsentativen Wettergradienten:

| Spot-Check | Observable | Eingangsgröße | Betriebspunkt |
|-----------|-----------|---------------|---------------|
| 1 | `vm_mv_bus_2_pu` | `g_poa_wm2` | `base` |
| 2 | `p_slack_mw` | `t_amb_c` | `base` |
| 3 | `total_p_loss_mw` | `g_poa_wm2` | `load_high` |
| 4 | `vm_mv_bus_2_pu` | `wind_ms` | `base` |

FD-Schrittweiten: `g_poa_wm2`: 5 W/m², `t_amb_c`: 0.1 °C, `wind_ms`: 0.1 m/s.

Ziel ist eine grobe Plausibilisierung der neuen Wetterkette, nicht eine
vollständige Gradientenvalidierung. Die Korrektheit des impliziten AD-Kerns wurde
bereits in Experiment 2 umfassend nachgewiesen.

## Artefakte

Ordner: `experiments/results/exp03_cross_domain_pv_weather/`

| Datei | Format | Beschreibung |
|-------|--------|--------------|
| `scenario_grid.csv/json` | tidy | Forward-Solve-Ergebnisse: eine Zeile pro (Szenario, Wetterfall, Observable) |
| `sensitivity_table.csv/json` | tidy | AD-Sensitivitäten: eine Zeile pro (Szenario, Wetterfall, Observable, Eingang) |
| `gradient_spotcheck.csv/json` | tidy | AD-vs-FD-Vergleich für 4 ausgewählte Gradienten |
| `run_summary.csv/json` | tidy | Zusammenfassung pro Betriebspunkt |
| `metadata.json` | JSON | Reproduzierbarkeitsdaten (Zeitstempel, Git-Hash, Parameter) |
| `README.md` | Text | Menschenlesbare Beschreibung der Artefakte |

### Tidy-Format-Regeln

- Eine Zeile = eine Beobachtung.
- Keine verschachtelten Listen in CSV-Zellen.
- `network_scenario`, `weather_case_id`, `weather_case_type`, `g_poa_wm2`,
  `t_amb_c`, `wind_ms`, `observable`, `input_parameter`, `value`, `unit` als
  eigene Spalten.

## Bewusste Vereinfachungen und Grenzen

| Vereinfachung | Begründung |
|---------------|------------|
| PQ-Einspeisung statt PV-Bus | Keine Spannungsregelung im aktuellen Modellscope |
| Keine Q-Limits | Nicht Teil des PoC-Scopes |
| Keine PV-PQ-Umschaltung | Keine Controller-Logik implementiert |
| Keine Controllerlogik | Außerhalb des Scopes |
| `alpha`, `kappa` fest | In Exp. 3 keine Variation; erst in Exp. 5 |
| `wind_adj = wind_ms` | Keine Höhen-/Montagekorrektur in V1 des NOCT-SAM |
| Wettergrößen außerhalb p.u. | Korrekt: meteorologische Einheiten bleiben erhalten |
| Kompakter Szenarioraum | 93 Solves statt eines großen 3D-Würfels |
| Kein vollständiger Wachstumsraum | Für PoC ausreichend; keine riesige Parameterexplosion |
| Nur `example_simple()` | Ab Exp. 3 ausschließlich dieser Demonstrator |

## Abgrenzung zu anderen Experimenten

### Abgrenzung zu Experiment 2

- Exp. 2 validiert den numerischen Gradienten-Kern (elektrische Parameter → Observables).
- Exp. 2 nutzt 48 systematische Gradienten mit FD-Schrittweitenstudie.
- **Exp. 3** fügt den meteorologischen Eingangsraum hinzu (Wetter → PV → Netz).
- Exp. 3 ersetzt den `sgen` durch ein wetterbetriebenes Modell.
- Exp. 3 nutzt nur einen gezielten Spot-Check, keine vollständige Schrittweitenstudie.

### Abgrenzung zu Experiment 5

- Exp. 5 optimiert eine elektrische Zielgröße durch Variation von Upstream-Variablen.
- In Exp. 5 sind `alpha`, `kappa` oder andere Upstream-Parameter die
  Optimierungsvariablen.
- **Exp. 3** analysiert nur Sensitivitäten (Vorwärts- und Rückwärtsdifferenzierung),
  führt aber keine Optimierungsschleife aus.

## Geplante spätere Visualisierungsideen (noch nicht umgesetzt)

Diese Visualisierungen sind als mögliche spätere Auswertungsschritte angedacht,
sind aber **kein Bestandteil des aktuellen Implementierungsschritts**:

- Kurven `t_amb_c -> vm_mv_bus_2_pu` für verschiedene Betriebspunkte (1D-Sweep).
- Heatmap `g_poa_wm2 × t_amb_c -> d_vm/d_g` (Sensitivitätsgitter aus dem 2D-Gitter).
- Vergleichsgrafik AD-Gradient vs. FD-Gradient für die vier Spot-Check-Fälle.
- Sensitivitätsbalkendiagramm: welche Wettergröße hat den stärksten Einfluss auf
  welches Observable?
