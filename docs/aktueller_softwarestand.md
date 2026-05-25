# Aktueller Softwarestand

Stand: 2026-05-20

Diese Datei fasst den aktuell umgesetzten Softwarestand knapp zusammen.

## Numerischer Kern

Der numerische Kern ist umgesetzt. Dazu gehören die zentralen Netz- und
Parameterdatentypen, die Per-Unit-Umrechnung, der Aufbau der Admittanzmatrix,
die Residuenformulierung, der Newton-Raphson-Solver, die implizite
Differentiation und die Auswertung elektrischer Observables.

Die Y-Bus-Stempelung umfasst Leitungen, Zweiwicklungs-Transformatoren und
Shunts. Der Kern wird für stationäre AC-Power-Flow-Rechnungen genutzt und ist
in den Experimenten sowie den zugehörigen Tests eingebunden.

## Netz- und Modellpipeline

Die Pipeline zum Einlesen und Aufbereiten von Netzen ist umgesetzt. JSON-Netze
und `pandapower`-Netze können in die interne Netzbeschreibung überführt,
validiert, kompiliert und anschließend vom numerischen Kern gelöst werden.

Für `pandapower` sind die im aktuellen Scope verwendeten Elemente angebunden,
einschließlich Busse, Slack, Lasten, SGen/Gen im scope-matched Modell,
Leitungen, Zweiwicklungs-Transformatoren, Shunts sowie der Vorverarbeitung von
Switches und Bus-Fusionen.

Die Kopplung vorgelagerter Modelle ist ebenfalls umgesetzt. PV-Modelle liefern
P/Q-Einspeisungen über eine gemeinsame Schnittstelle in die Netzparameter. Dazu
gehören das analytische PV-Wettermodell, die NOCT-SAM-Zelltemperaturkopplung,
die direkte P/Q-Baseline sowie das neuronale P-only-Surrogat mit fester
Q/P-Kopplung.

## Umgesetzte Experimente

Experiment 1 ist umgesetzt. Es validiert den Vorwärtslauf gegen `pandapower`,
einschließlich des ursprünglichen 3-Bus-PoC und der `example_simple()`-Variante
mit scope-matched Referenz.

Experiment 2 ist umgesetzt. Es validiert Gradienten des implizit
differenzierten Power-Flow-Kerns gegen Finite Differences und enthält eine
Schrittweitenanalyse.

Experiment 3 ist umgesetzt. Es koppelt Wettergrößen über ein PV-Modell an den
AC-Power-Flow und berechnet Forward-Ergebnisse sowie End-to-End-Sensitivitäten
für elektrische Observables.

Experiment 4 ist umgesetzt. Es demonstriert die modulare Kopplung verschiedener
vorgelagerter Modelle, einschließlich analytischem PV-Wettermodell, direkter
P/Q-Baseline und neuronalen NN-Surrogats. Training, Modellvergleich,
Power-Flow-Vergleich und Sensitivitätsauswertung werden als Artefakte
exportiert.

Experiment 5 ist umgesetzt. Es besteht aus dem Netz-Screening für PV-Fälle
und einer anschließenden eindimensionalen PV-Curtailment-Optimierung für den
ausgewählten Demonstratorfall.

## Artefaktstand

Die Experimente schreiben ihre Ergebnisse in strukturierte Ergebnisordner unter
`experiments/results/`. Dort liegen die jeweiligen CSV-/JSON-Artefakte,
Metadaten, README-Dateien und, soweit umgesetzt, Abbildungen.
