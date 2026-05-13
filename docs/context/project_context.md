# Projektkontext: `diffpf`

## Ziel

`diffpf` ist ein Python-/JAX-Projekt zur Entwicklung eines differenzierbaren stationären AC-Power-Flow-Kerns. Ziel ist nicht die Entwicklung eines neuen Lastflussverfahrens und nicht der vollständige Ersatz etablierter Werkzeuge wie `pandapower`, sondern der Nachweis, dass elektrische Netzphysik als differenzierbare Rechenschicht in größere Modellketten eingebettet werden kann.

Der Kernbeitrag liegt in der End-to-End-Differenzierbarkeit über Modellgrenzen hinweg:

```text
vorgelagertes Modell -> elektrische P/Q-Injektion -> AC-Power-Flow -> Netzgröße
```

Dadurch werden Sensitivitäten nicht-elektrischer Eingangsgrößen, etwa Einstrahlung, Temperatur oder Modellparameter, auf elektrische Zielgrößen wie Knotenspannungen, Slack-Leistung, Transformatorflüsse oder Verluste direkt berechenbar.

## Wissenschaftlicher Beitrag

Die Arbeit ist ein Proof of Concept im Bereich Differentiable Physics, Scientific Machine Learning und stationärer elektrischer Netzberechnung. Der Beitrag besteht aus:

- einem modularen AC-Power-Flow-Kern in JAX,
- einer klaren Trennung zwischen Netz-I/O, Kompilierung und numerischem Kern,
- impliziter Differentiation durch das gelöste Lastflussproblem,
- einer kontrollierten `pandapower`-I/O-Pipeline,
- Validierung von Vorwärtssolve und Gradienten,
- Vorbereitung und Umsetzung der Kopplung vorgelagerter Modelle.

Die Arbeit zeigt nicht, dass `diffpf` besser als klassische Power-Flow-Tools ist. Sie zeigt, dass ein stationäres Netzmodell so formuliert werden kann, dass es in differenzierbaren Modellketten nutzbar wird.

## Scope

### Im aktuellen Scope

- stationärer symmetrischer AC-Power-Flow,
- Slack-Bus,
- PQ-Busse,
- feste P/Q-Einspeisungen und Lasten,
- Leitungen im Pi-Ersatzschaltbild,
- 2-Wicklungs-Transformatoren,
- Shunts,
- einfache Switch-Topologie,
- `pandapower`-Import für den unterstützten Modellumfang,
- implizite Gradienten gegen Netz- und Einspeiseparameter,
- vorgelagerte, JAX-kompatible P/Q-Modelle.

### Nicht im aktuellen Scope

- vollständige `pandapower`-Kompatibilität,
- industrielle Großnetzberechnung,
- dreiphasige oder unsymmetrische Lastflüsse,
- Controllerlogiken,
- Schutz- und Schaltlogik,
- 3-Wicklungs-Transformatoren,
- `ward`, `xward`, `impedance`, `dcline`,
- Generator-Q-Limits,
- PV↔PQ-Umschaltung,
- vollständige dynamische Netzsimulation.

## Hauptdemonstrator

Der aktuelle Hauptdemonstrator ist das `pandapower`-Netz `example_simple()`.

Dieses Netz enthält:

- 110-kV- und 20-kV-Spannungsebenen,
- einen `ext_grid` als Slack,
- einen 110/20-kV-Transformator,
- mehrere Leitungen,
- geschlossene Bus-Bus-Switches,
- offene und geschlossene Line-Switches,
- eine Last am Bus `"MV Bus 2"`,
- einen `gen` am Bus `"MV Bus 1"`,
- einen `sgen` mit Name `"static generator"` am Bus `"MV Bus 2"`,
- einen Shunt.

Für die Kopplung vorgelagerter PV-Modelle ist festgelegt:

```text
Kopplungspunkt:     "MV Bus 2"
Ersetztes Element:  sgen "static generator"
Referenzwerte:      P = 2.0 MW, Q = -0.5 MVAr
Modellierung:       wetterabhängige P/Q-Injektion, kein spannungsregelnder PV-Bus
```

Der ursprüngliche 3-Bus-PoC bleibt als historischer Minimal- und Kontrollfall erhalten. Für neue Experimente ab Experiment 3 wird jedoch ausschließlich das `pandapower`-Netz `example_simple()` verwendet, angepasst an die jeweilige Fragestellung.

## Daten- und Modellpipeline

Die zentrale Zielpipeline lautet:

```text
pandapower.to_json
    -> RawPandapowerNetwork / RawNetwork
    -> kanonische RAW-Zwischenrepräsentation
    -> NetworkSpec
    -> CompiledTopology + NetworkParams + PFState
    -> JAX-Core
```

`pandapower` bleibt Eingabe-, Referenz- und Validierungswerkzeug. Der JAX-Kern bleibt unabhängig von `pandapower`.

## Experimentelle Strategie

Die Experimente bauen aufeinander auf:

1. Vorwärtsvalidierung gegen `pandapower`, inklusive `example_simple()`.
2. Gradientenvalidierung AD/implizit gegen zentrale Finite Differences, inklusive `example_simple()`.
3. Cross-Domain-Sensitivität mit vorgelagertem PV-Modell am `example_simple()`-Netz.
4. Modularität der Kopplung verschiedener Upstream-Modelle am `example_simple()`-Netz.
5. Einfache gradientenbasierte gekoppelte Optimierung am `example_simple()`-Netz.
6. Struktur- und I/O-Validierung der `pandapower`-Pipeline.

## Projektstatus

Das Fundament ist gelegt:

- Der numerische V1-Kern ist implementiert.
- Die `pandapower`-I/O-Pipeline ist umgesetzt.
- `example_simple()` wird erfolgreich importiert, kompiliert und gelöst.
- Experiment 1b validiert den Vorwärtssolve im erweiterten Netz.
- Experiment 2b validiert 48 Gradienten im erweiterten Netz.
- Der PV-Kopplungspunkt an `"MV Bus 2"` ist festgelegt.
- Ein JAX-kompatibles PV-P/Q-Kopplungsinterface ist vorbereitet.

Der nächste fachliche Schwerpunkt ist die Integration vorgelagerter Modelle ab Experiment 3.
