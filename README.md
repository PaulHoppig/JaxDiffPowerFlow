# diffpf

**Differenzierbarer AC-Power-Flow-Kern in Python/JAX zur Kopplung elektrischer Netzphysik mit vorgelagerten Modellen**

`diffpf` ist ein Proof of Concept für einen modularen, differentiierbaren stationären AC-Leistungsflusskern in JAX. Ziel ist **nicht** der Aufbau eines vollständigen industriellen Netzsimulators, sondern einer sauberen, wissenschaftlich nachvollziehbaren Physikschicht, die in größere differentiierbare Rechengraphen eingebettet werden kann – zum Beispiel:

**Wetter → PV-Modell → PV-Einspeisung → AC-Netzmodell → Spannungen / Verluste / Netzbezug**

Der methodische Mehrwert liegt in der **modellübergreifenden Differenzierbarkeit**: Sensitivitäten sollen nicht nur nach elektrischen Eingangsgrößen, sondern auch nach **vorgelagerten, nicht-elektrischen Größen** wie Einstrahlung oder Temperatur berechenbar werden.

---

## Motivation

Klassische Power-Flow-Modelle sind zentral für Analyse, Planung und Betrieb elektrischer Netze. In vielen modernen Anwendungen hängen netzseitige Zielgrößen jedoch nicht isoliert von elektrischen Sollwerten ab, sondern von **gekoppelten vorgelagerten Modellen**, zum Beispiel aus Wetter, Erzeugung, Last oder datengetriebenen Ersatzmodellen.

`diffpf` adressiert genau diese Schnittstelle:

- stationäre AC-Netzphysik bleibt explizit modelliert,
- der Power-Flow-Kern wird als **differenzierbare Rechenschicht** formuliert,
- Gradienten können über Modellgrenzen hinweg propagiert werden,
- der numerische Netzkern bleibt dabei möglichst **modular und unverändert wiederverwendbar**.

---

## Projektziel

Die Arbeit soll zeigen, dass ein AC-Power-Flow-Kern in JAX

1. elektrisch korrekt rechnet,
2. lokal korrekt differenzierbar ist,
3. mit unterschiedlichen vorgelagerten Modellen gekoppelt werden kann,
4. und dadurch modellübergreifende Sensitivitäts- und einfache Optimierungsaufgaben unterstützt.

---

## Was das Projekt ist – und was nicht

### Das Projekt ist

- ein **differenzierbarer AC-Power-Flow-Demonstrator** in Python/JAX,
- eine **physikalische Rechenschicht** für gekoppelte Modellketten,
- eine Grundlage für **Cross-Domain-Sensitivitätsanalysen**,
- ein methodischer Ausgangspunkt für spätere gekoppelte Optimierungs- oder Lernaufgaben.

### Das Projekt ist nicht

- **kein** neues oder überlegenes elektrisches Lastflussverfahren,
- **kein** vollständiges Netzberechnungsframework,
- **kein** Nachweis allgemeiner Echtzeitfähigkeit,
- **keine** allgemeine Überlegenheit gegenüber klassischen PF-/OPF-Ansätzen,
- **keine** Untersuchung großer realer Netze oder diskreter Regelungslogiken.

---

## Methodischer Kern

Das Projekt betrachtet Modellketten der Form:

```text
vorgelagertes Modell -> elektrische Einspeisung/Last -> AC-Power-Flow -> netzseitige Zielgröße
```

Der stationäre Betriebspunkt wird durch einen Power-Flow-Solver bestimmt. Die Sensitivitäten werden **am gelösten Betriebspunkt** über **implizite Differenzierung** berechnet, statt durch das vollständige Unrolling aller Newton-Schritte.

### Mathematische Grundform

Die Physik basiert auf der komplexen AC-Formulierung:

```text
I = Y_bus @ V
S_calc = V * conj(I)
```

Daraus werden die Residuen gebildet:

```text
r_P = P_spec - Re(S_calc)
r_Q = Q_spec - Im(S_calc)
```

Zusätzlich wird eine Residual-Loss der Form

```text
0.5 * ||r||^2
```

verwendet.

### Methodische Fixpunkte

Die folgenden Designentscheidungen sind im Projekt bewusst gesetzt:

- interne Rechnung konsequent im **Per-Unit-System**,
- elektrische Physik in **komplexer Form**,
- freie Solverzustände **reell parametrisiert**,
- Spannung in **rechteckiger Darstellung**: `V = Vr + j Vi`,
- Solverzustand der Nicht-Slack-Busse als `[Vr | Vi]`,
- stationärer Forward-Solve über **Newton-Raphson**,
- Gradienten über den konvergierten Solver via **implizite Differenzierung** mit `jax.lax.custom_root`,
- Trennung zwischen **statischer Topologie**, **differenzierbaren Parametern** und **freien Zuständen**.

---

## Scope von V1

Die erste Version fokussiert auf einen bewusst klar abgegrenzten Kern.

### Unterstützt

- beliebige Netze aus
  - Bussen,
  - verlustbehafteten Leitungen im **Pi-Ersatzschaltbild**,
  - bekannten Knotenleistungen,
- **einen Slack-Bus**,
- **PQ-Busse**,
- stationäre AC-Physik,
- Per-Unit-Rechnung,
- JAX-kompatible Datenstrukturen,
- differentiierbare Parameter und Zustände.

### Explizit nicht Teil von V1

- Transformatoren,
- PV-Bus-Gleichungen,
- große Benchmark-Netze,
- industrielle Vollständigkeit,
- harte Echtzeitanwendungen,
- nichtglatte oder diskrete Regellogiken.

---

## Demonstrator

Als Proof of Concept dient ein kleines **3-Bus-Verteilnetz**:

- **Bus 0**: Slack / Umspannwerk / Anbindung ans übergeordnete Netz
- **Bus 1**: Wohngebiet / Last
- **Bus 2**: PV-Park

Das grundlegende Demonstrationsnarrativ lautet:

Ein Wohngebiet wird lokal durch einen PV-Park versorgt und zusätzlich über ein Umspannwerk an das übergeordnete Netz angebunden. Ein vorgelagertes differentiierbares PV-Modell erzeugt aus Wettergrößen wie Einstrahlung und Temperatur eine Einspeiseleistung, die anschließend in den differentiierbaren Netzrechenkern eingeht.

Typische Ausgänge des Netzkerns sind:

- Netzbezug am Slack-Bus (`P_grid`),
- Spannungsbeträge `|V|`,
- Spannungswinkel,
- Netzverluste,
- Leitungsflüsse,
- Residuum / Residual-Loss.

---

## Softwarearchitektur

Eine zentrale Architekturentscheidung ist die **zweistufige Repräsentation**:

- **nach außen**: menschenfreundliche, deklarative Eingabestrukturen,
- **nach innen**: kompilierte, arraybasierte JAX-Repräsentation.

Damit bleibt die Nutzerschnittstelle lesbar, während der numerische Kern JIT- und Autodiff-freundlich bleibt.

### Kernprinzipien

- `core/` enthält **nur** numerische Kernlogik,
- `io/` und `parser/` kapseln Einlesen, Validierung und Mapping,
- `compile/` überführt Eingabemodelle in JAX-taugliche interne Strukturen,
- `solver/` kapselt Forward-Solve und implizite Differenzierung,
- `validation/` enthält Referenzvergleiche und Finite-Difference-Checks,
- `experiments/` bildet die wissenschaftlichen Experimente ab,
- `tests/` sichert Kernlogik, Parser, Solver und Observables ab.

### Projektstruktur

```text
src/diffpf/
├── io/
│   ├── reader.py
│   └── parser.py
├── core/
│   ├── types.py
│   ├── units.py
│   ├── ybus.py
│   ├── residuals.py
│   └── observables.py
├── compile/
│   └── network.py
├── solver/
│   ├── newton.py
│   └── implicit.py
└── validation/
    ├── finite_diff.py
    ├── pandapower_ref.py
    └── gradient_check.py

cases/
experiments/
tests/
```

### Wichtige interne Datentypen

- `NetworkSpec`
- `CompiledTopology`
- `NetworkParams`
- `PFState`

Dabei gilt:

- **Topologie** ist statisch,
- **physikalische Parameter** sind differentiierbar,
- **Zustände** sind solverseitig frei.

---

## Aktueller Entwicklungsstand

Laut Projektstatus und Changelog sind bereits folgende Bausteine umgesetzt:

### Numerischer Kern

- JAX-kompatible Typen und Pytrees,
- Per-Unit-Konvertierung,
- `Y_bus`-Aufbau für Pi-Leitungen,
- Residuenformulierung,
- Residual-Loss,
- Newton-Raphson-Solver,
- implizite Differenzierung via `custom_root`,
- solver-unabhängige Observables.

### Parser- und IO-Schicht

- JSON-Loader,
- semantische Netzvalidierung,
- Mapping externer Bus-IDs auf interne Indizes,
- Umrechnung physikalischer Größen in p.u.,
- Aufbau von `CompiledTopology`, `NetworkParams` und Startzustand.

### Validierung und Experimente

- Referenzvergleich gegen **pandapower**,
- zentrale Finite Differences,
- Gradient-Checks AD vs. FD,
- wiederverwendbare Betriebspunkte `low_pv`, `medium_pv`, `high_pv`,
- Tests für Parser, Newton-Solver, impliziten Solver, Observables und Validierung.

---

## Wissenschaftliche Evaluation

Die geplante Evaluation folgt einer gestuften Logik: Jede stärkere Aussage baut auf einer vorher abgesicherten Grundlage auf.

### Experiment 1 – Solver-Validierung

Nachweis, dass der JAX-Kern für repräsentative Betriebspunkte dieselben elektrischen Ergebnisse liefert wie ein Referenzsolver.

### Experiment 2 – Gradientenvalidierung

Vergleich von Automatic Differentiation / impliziter Differenzierung mit zentralen Finite Differences, um die lokale Konsistenz der berechneten Sensitivitäten zu prüfen.

### Experiment 3 – Cross-Domain-Sensitivitätsanalyse

Demonstration, dass netzseitige Zielgrößen bis in vorgelagerte nicht-elektrische Eingangsgrößen zurückdifferenziert werden können.

### Experiment 4 – Modularität der Modellkopplung

Nachweis, dass derselbe PF-Kern mit unterschiedlichen vorgelagerten Modellen gekoppelt werden kann, ohne den Kern selbst anzupassen.

### Experiment 5 – Gekoppelte Optimierung

Demonstration einfacher gradientenbasierter Optimierungsaufgaben über die gesamte Modellkette hinweg.

---

## Bereits erreichte Ergebnisse

Nach aktuellem Changelog sind insbesondere folgende Punkte bereits realisiert:

- **Experiment 1**: Validierung gegen `pandapower` für das 3-Bus-PoC-Netz; laut Status stimmen die Ergebnisse innerhalb numerischen Rundungsrauschens überein.
- **Experiment 2**: lokale Validierung impliziter Gradienten gegen zentrale Finite Differences.
- Implementierte Ergebnisartefakte für die Gradientenvalidierung, u. a. Tabellen zu Fehlern und Schrittweitenstudien.

Diese Ergebnisse stützen bereits zwei zentrale Aussagen des Projekts:

1. der Kern rechnet elektrisch konsistent,
2. die abgeleiteten lokalen Sensitivitäten sind numerisch überprüfbar.

---

## Designregeln für die Weiterentwicklung

Für weitere Arbeit am Projekt gelten einige bewusst strenge Regeln:

- keine stillen Änderungen an Residuenformulierung oder Koordinatenwahl,
- kein `numpy` im Hot Path – dort ausschließlich `jax.numpy`,
- keine Python-`if`-Abfragen auf Tracer-Werten in `core/`, `solver/` oder `pipeline/`,
- keine globalen Zustände im numerischen Kern,
- differentiierbare Parameter gehören in `NetworkParams`, nicht in versteckte Closures,
- `pandapower` bleibt Referenz und darf nicht in den Kern importiert werden,
- neue Abstraktionen erst **nach** Wiederverwendung vorhandener Typen und Funktionen,
- neue Kernfunktionen nur zusammen mit Tests.

---

## Grenzen

Die Aussagen dieses Projekts sind bewusst **lokal und demonstratorbasiert**.

Die Validierung ist **kein** allgemeiner mathematischer Beweis für beliebige Netze oder Betriebspunkte. Sie ist auch **kein** Nachweis für Skalierbarkeit auf große reale Systeme, nicht für diskrete Regellogiken und nicht für allgemeine betriebliche Echtzeitanwendungen.

---

## Anschlussfähigkeit

Das Projekt ist als Grundlage für weiterführende Arbeiten gedacht, zum Beispiel für:

- größere Netze,
- Zeitreihen- und Multi-Betriebspunkt-Analysen,
- mehrere gekoppelte Domänenmodelle,
- modellübergreifende Optimierung,
- datengetriebene Ersatzmodelle,
- Einbettung von Netzphysik in lernbasierte Verfahren.

---

## Hinweise zum Repository-Setup

Dieser README-Entwurf basiert auf dem inhaltlichen Projektkontext und dem dokumentierten Softwarestand. Da in den vorliegenden Unterlagen **keine verifizierten Installations- oder CLI-Kommandos** enthalten sind, wurden bewusst **keine erfundenen Setup-Befehle** aufgenommen.

Sobald im Repository z. B. eine `pyproject.toml`, `requirements.txt`, `Makefile` oder konkrete Einstiegsskripte vorliegen, sollten folgende Abschnitte ergänzt werden:

- **Installation**
- **Quickstart / Minimal Example**
- **Tests ausführen**
- **Experimente reproduzieren**

---

## Status

**Projektphase:** Proof of Concept / wissenschaftlicher Demonstrator  
**Fokus:** Differentiable Physics für stationären AC-Power-Flow in JAX  
**Aktuell umgesetzt:** Kernsolver, Parser, Referenzvalidierung, implizite Gradientenvalidierung  
**Nächste fachliche Schritte:** Cross-Domain-Sensitivitäten, Modularitätsnachweis, gekoppelte Optimierung

