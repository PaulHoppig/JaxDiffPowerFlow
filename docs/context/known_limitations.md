# Bekannte Grenzen und Future Work

## Grundsätzliche Einordnung

`diffpf` ist ein wissenschaftlicher Demonstrator für differenzierbare AC-Netzphysik, kein vollständiges Netzberechnungswerkzeug. Die aktuellen Grenzen sind bewusst dokumentiert und müssen in Experimenten und Auswertung berücksichtigt werden.

## Keine vollständige `pandapower`-Kompatibilität

Die `pandapower`-I/O-Pipeline unterstützt einen kontrollierten Teilumfang. Nicht unterstützte aktive Elemente sollen nicht still ignoriert, sondern klar abgewiesen werden.

Nicht unterstützt:

- 3-Wicklungs-Transformatoren,
- `ward`,
- `xward`,
- `impedance`,
- `dcline`,
- Controller,
- vollständige Schaltlogiken,
- unsymmetrische / dreiphasige Lastflüsse.

## Generator- und PV-Bus-Semantik

Ein `pandapower.gen` ist im originalen `pandapower`-Lastfluss ein spannungsregelnder PV-Bus:

```text
gegeben: P und |V|
gesucht: Q
```

Im aktuellen validierten `diffpf`-Scope wird `gen` nicht als vollständiger Spannungsregler behandelt. Im `scope_matched`-Vergleich wird er als feste P-Einspeisung mit `Q=0` modelliert.

Nicht umgesetzt:

- aktive Spannungsregelung für `gen` im `pandapower`-Adapter,
- Generator-Q-Limits,
- PV↔PQ-Umschaltung bei Q-Grenzverletzung.

Konsequenz: Der `original_pandapower`-Vergleich ist nur ein Kontextvergleich, kein strikter Gleichheitstest.

## PV-Anlage ist kein PV-Bus

Für die geplante Upstream-Kopplung wird die PV-Anlage als wetterabhängige P/Q-Einspeisung modelliert. Der Kopplungsbus `"MV Bus 2"` bleibt PQ-Bus. Diese Entscheidung ist bewusst, weil dadurch der Rechengraph glatt und direkt differenzierbar bleibt.

## Transformator-Modellparität

`diffpf` unterstützt 2-Wicklungs-Transformatoren mit Pi-Modell, Tap und Phasenverschiebung. Im `example_simple()`-Netz bleibt jedoch ein systematischer Offset bei Transformatorflüssen und Verlusten gegenüber `pandapower`.

Aktueller Befund:

- Knotenspannungen stimmen im `scope_matched`-Modus sehr gut überein.
- Leitungsverluste stimmen sehr gut überein.
- Trafoverluste, Slack-P/Q und Trafoflüsse zeigen einen systematischen Offset.

Naheliegende Ursache ist eine nicht vollständig identische Trafo-Stempelung bzw. Verlust-/Shift-Behandlung, besonders wegen der 150°-Phasenverschiebung des Transformators in `example_simple()`.

Future Work:

- isolierter 2-Bus-Trafo-Test gegen `pandapower`,
- Varianten mit `shift = 0°`, `30°`, `150°`,
- Varianten mit/ohne Magnetisierung,
- Varianten mit/ohne Tap,
- Prüfung von Tap-Seite und Vorzeichenkonvention.

## Initialisierung

Flat Start kann bei großen Transformator-Phasenverschiebungen divergieren. Für `example_simple()` wird deshalb eine trafo-shift-aware Initialisierung verwendet. `pandapower`-Referenzläufe verwenden `init="dc"`.

Das ist kein Modellfehler, sondern eine Konvergenzmaßnahme. Die Bewertung muss auf dem konvergierten Residuum erfolgen.

## Statische Topologie

Die Topologie ist im JAX-Kern statisch.

Nicht differenzierbar sind:

- Switch-Zustände,
- Bus-Fusion,
- Element aktiv/inaktiv,
- diskrete Controllerzustände,
- PV↔PQ-Umschaltung.

Topologieänderungen müssen vor dem Compile-Schritt erfolgen.

## Vereinfachtes Switch-Handling

- Geschlossene Bus-Bus-Switches werden als Bus-Fusion behandelt.
- Offene Line-Switches deaktivieren die gesamte Leitung.
- Detaillierte offene Leitungsenden werden nicht modelliert.

Diese Vereinfachung ist für den aktuellen Demonstrator ausreichend, aber nicht vollständig `pandapower`-äquivalent.

## Leitungs- und Shunt-Vereinfachungen

Leitungen werden als Pi-Modell abgebildet. Bestimmte Detailparameter wie Leitwertbeläge werden aktuell nicht vollständig berücksichtigt, sofern sie nicht explizit in die unterstützte p.u.-Konvertierung eingebunden sind.

Shunts sind konstante Admittanzen ohne Regelung.

## Numerische und wissenschaftliche Grenzen

Die Validierung ist lokal und demonstratorbezogen. Sie ist kein mathematischer Beweis für beliebige Netze oder Betriebspunkte.

Die Gradientenvalidierung zeigt numerische Konsistenz gegenüber Finite Differences im untersuchten Parameterraum. Sie garantiert nicht automatisch Robustheit für:

- schlecht konditionierte Netze,
- Spannungsinstabilität,
- diskrete Regelwechsel,
- sehr große Netze,
- stark nichtlineare oder nichtglatte Upstream-Modelle.

## Nicht-Ziele der Bachelorarbeit

Nicht Ziel der aktuellen Arbeit sind:

- vollständiger OPF,
- vollständige Controllerintegration,
- probabilistische Simulation,
- Echtzeitfähigkeit,
- industrielle Netzgröße,
- vollständiges `pandapower`-Feature-Matching,
- allgemeine Überlegenheit gegenüber klassischen PF/OPF-Verfahren.

## Priorisierte Future Work

1. End-to-End-Upstream-Experimente auf `example_simple()` abschließen.
2. PV-Bus-Enforcement für idealisierte `gen`-Semantik sauber validieren.
3. Transformator-Stempel isoliert gegen `pandapower` feinabgleichen.
4. Q-Limit-Handling und PV↔PQ-Umschaltung als äußere, diskrete Logik untersuchen.
5. Weitere kleine `pandapower`-Netze als Regressionstests ergänzen.
6. Größere Netze erst nach Stabilisierung der Modellsemantik prüfen.
