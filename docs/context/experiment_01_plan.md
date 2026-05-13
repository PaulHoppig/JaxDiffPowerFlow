# Experiment 1 - Detaillierter Plan: example_simple() Solver-Validierung

## Wissenschaftliches Ziel

Experiment 1 validiert den stationaeren AC-Power-Flow-Kern von `diffpf`
gegen `pandapower` auf dem erweiterten Hauptdemonstrator
`pandapower.networks.example_simple()`. Ziel ist der Nachweis, dass der
JAX-Solver im dokumentierten Modellscope konsistente Knotenspannungen,
Winkel, Slack-Leistungen, Leitungsfluesse, Trafofluesse und Verluste liefert.

Der Plan beschreibt nur die Experimentdurchfuehrung auf `example_simple()`.
Der historische 3-Bus-PoC bleibt ausserhalb dieses Detailplans.

## Demonstrator und Modellscope

- **Netz:** `pandapower.networks.example_simple()`.
- **Spannungsebenen:** 110-kV-HV-Ebene und 20-kV-MV-Ebene.
- **Besonderheit:** Zwei geschlossene Bus-Bus-Schalter werden vor dem Solve
  zu internen Bussen fusioniert.
- **Trafo:** Zweiwicklungs-Transformator mit grosser Phasenverschiebung
  von 150 Grad.
- **Kopplungsbus fuer Folgeexperimente:** `"MV Bus 2"`.
- **Referenz-sgen fuer Folgeexperimente:** `sgen "static generator"`.

Der `diffpf`-Kern behandelt aktuell keine spannungsregelnden PV-Busse mit
Q-Limits. Deshalb wird fuer die strikte Validierung ein scope-matched Modell
verwendet, in dem aktive `gen`-Elemente als `sgen(P=gen.p_mw, Q=0)` modelliert
werden.

## Referenzmodi

| Modus | Bedeutung | Bewertung |
|-------|-----------|-----------|
| `scope_matched` | Aktive `gen` werden in `sgen(P, Q=0)` konvertiert; `pandapower` und `diffpf` loesen dasselbe PQ-Modell. | Strikte numerische Validierung |
| `original_pandapower` | `pandapower` loest das Originalnetz mit PV-Bus-Generator; `diffpf` bleibt im PQ-Scope. | Kontextvergleich, keine Gleichheitsforderung |

Nur `scope_matched` ist ein harter Paritaetstest. Im
`original_pandapower`-Modus sind Abweichungen bei Spannung und Blindleistung am
Generatorbus erwartbar, weil `pandapower` dort echte Generator-
Spannungsregelung modelliert.

## Elektrische Betriebspunkte

| Szenario | Lastfaktor | sgen-Faktor | Zweck |
|----------|------------|-------------|-------|
| `base` | 1.00 | 1.00 | Nominalzustand |
| `load_low` | 0.75 | 1.00 | Reduzierte Last |
| `load_high` | 1.25 | 1.00 | Erhoehte Last |
| `sgen_low` | 1.00 | 0.50 | Reduzierte statische Einspeisung |
| `sgen_high` | 1.00 | 1.50 | Erhoehte statische Einspeisung |
| `combined_high_load_low_sgen` | 1.25 | 0.50 | Hohe Last, niedrige Einspeisung |
| `combined_low_load_high_sgen` | 0.75 | 1.50 | Niedrige Last, hohe Einspeisung |

Die Lastvariation erfolgt ueber das `scaling`-Feld der vorhandenen Last. Die
Einspeisevariation erfolgt ueber das `scaling`-Feld des vorhandenen statischen
Generators. Die Topologie bleibt fuer alle Szenarien konstant.

## Solver- und Referenzeinstellungen

`diffpf` verwendet den Newton-Solver mit:

| Option | Wert |
|--------|------|
| `max_iters` | 50 |
| `tolerance` | `1e-10` |
| `damping` | `0.7` |
| Initialisierung | `trafo_shift_aware` |

`pandapower.runpp(...)` verwendet:

| Option | Wert |
|--------|------|
| `algorithm` | `nr` |
| `calculate_voltage_angles` | `True` |
| `init` | `dc` |
| `tolerance_mva` | `1e-9` |
| `max_iteration` | 50 |
| `trafo_model` | `pi` |
| `numba` | `False` |

Die trafo-shift-aware Initialisierung setzt HV-Busse auf den Slack-Winkel und
LV-Busse auf `slack_angle - trafo_shift_deg`. Sie ist fuer `example_simple()`
notwendig, weil ein Flat Start bei der 150-Grad-Phasenverschiebung divergieren
kann.

## Pflichtvergleiche

| Vergleich | Einheit | Beschreibung |
|-----------|---------|--------------|
| `vm_pu` | p.u. | Spannungsbetrag je Bus |
| `va_degree` | Grad | Spannungswinkel je Bus |
| `p_slack_mw` | MW | Slack-Wirkleistung |
| `q_slack_mvar` | MVAr | Slack-Blindleistung |
| `line_p_mw`, `line_q_mvar` | MW/MVAr | Leitungsfluesse je Richtung |
| `trafo_p_mw`, `trafo_q_mvar` | MW/MVAr | Trafofluesse HV/LV |
| `total_p_loss_mw` | MW | Gesamtwirkleistungsverluste |
| `total_q_loss_mvar` | MVAr | Gesamtblindleistungsverluste |
| `diffpf_residual_norm` | p.u. | Residualnorm am Loesungspunkt |

Zusaetzlich wird eine Strukturzusammenfassung exportiert, um Bus-Fusion,
Switch-Verarbeitung, aktive Leitungen, aktive Trafos, Shunts, Lasten und
Einspeisungen nachvollziehbar zu dokumentieren.

## Erwartete Ergebnisbewertung

Alle sieben Szenarien sollen in beiden Referenzmodi konvergieren. Im
`scope_matched`-Modus sind die Knotenzustaende der zentrale Validierungsanker:
Spannungsbetraege und Winkel sollen sehr nah an `pandapower` liegen.

Die gelieferten Artefakte zeigen fuer den Base-Fall im `scope_matched`-Modus
unter anderem:

| Kennzahl | Groessenordnung |
|----------|-----------------|
| `diffpf_residual_norm` | ca. `4e-11` |
| `max_vm_pu_abs_diff` | ca. `6e-5 p.u.` |
| `max_va_degree_abs_diff` | ca. `0.0023 deg` |
| `total_p_loss_mw_abs_diff` | ca. `0.014 MW` |
| `total_q_loss_mvar_abs_diff` | ca. `0.029 MVAr` |

Der verbleibende Offset in Trafofluesse, Slack-Leistung und Verlusten wird als
bekannte Modellgrenze dokumentiert. Die naheliegende Ursache ist eine nicht
vollstaendig `pandapower`-identische Trafo-Stempelung beziehungsweise
Verlust- und Shift-Behandlung.

## Artefakte

Ordner: `experiments/results/exp01_example_simple_validation/`

| Datei | Format | Beschreibung |
|-------|--------|--------------|
| `validation_summary.csv/json` | tidy | Eine Zeile pro Szenario und Referenzmodus; Konvergenz, Iterationen, Residualnorm und Fehlerkennzahlen |
| `bus_results.csv/json` | tidy | Spannungsbetrag und Winkel je Bus im Vergleich `diffpf` gegen `pandapower` |
| `slack_results.csv/json` | tidy | Slack-Wirk- und Blindleistung |
| `line_flows.csv/json` | tidy | Leitungsfluesse, Richtungen und Leitungsverluste |
| `trafo_flows.csv/json` | tidy | Trafofluesse auf HV- und LV-Seite sowie Trafoverluste |
| `losses.csv/json` | tidy | Gesamt-, Leitungs- und Trafoverluste |
| `structure_summary.csv/json` | tidy | Netzstruktur nach Switch-Verarbeitung und Bus-Fusion |
| `metadata.json` | JSON | Reproduzierbarkeitsdaten, Solveroptionen, Referenzmodi und bekannte Vereinfachungen |
| `README.md` | Text | Menschenlesbare Beschreibung der Ergebnisdateien |

## Tidy-Format-Regeln

- Eine Zeile beschreibt eine Beobachtung oder einen Vergleich.
- `scenario` und `reference_mode` sind in allen fachlichen Ergebnisdateien
  explizite Spalten.
- Bus-, Leitungs- und Trafo-IDs werden so exportiert, dass die Zuordnung zur
  `pandapower`-Quelle nachvollziehbar bleibt.
- CSV-Dateien enthalten keine verschachtelten Listen in Zellen.

## Bewusste Vereinfachungen und Grenzen

| Vereinfachung | Begruendung |
|---------------|------------|
| `gen -> sgen(P, Q=0)` im strikten Modus | Aktueller `diffpf`-Scope enthaelt keine PV-Bus-Spannungsregelung |
| Keine Q-Limits | Nicht Teil des validierten Kernscopes |
| Keine PV-PQ-Umschaltung | Keine Controllerlogik implementiert |
| `original_pandapower` nur Kontextvergleich | Unterschiedliche Generatorsemantik zwischen `pandapower` und `diffpf` |
| Trafo-Paritaet nicht perfekt | Bekannte Grenze bei 150-Grad-Phasenschieber und Verlustmodell |
| Shunt- und Switch-Verarbeitung nur im unterstuetzten Teilumfang | Kontrollierter Importumfang statt vollstaendiger `pandapower`-Reimplementierung |

## Abgrenzung zu Experiment 2

Experiment 1 validiert ausschliesslich den Forward-Solve und die elektrischen
Ergebnisgroessen gegen `pandapower`. Es berechnet keine AD-Gradienten und keine
Finite-Difference-Sensitivitaeten.

Experiment 2 baut auf diesem validierten `scope_matched`-Modell auf und prueft
dann lokale implizite Gradienten gegen zentrale finite Differenzen.

## Umgesetzte Visualisierung der Artefakte

Fuer Experiment 1 existiert aktuell keine separate Figurenpipeline. Die
Pflichtartefakte sind die CSV-/JSON-Tabellen und die menschenlesbare
`README.md`. Eine spaetere Visualisierung koennte Paritaetsplots fuer
Knotenspannungen, Slack-Leistungen oder Verlustkomponenten aus den vorhandenen
Artefakten erzeugen, ohne neue Power-Flow-Solves zu starten.
