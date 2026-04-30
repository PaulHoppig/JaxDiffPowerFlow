# Softwarekontext: `diffpf`

## Zweck

Diese Datei beschreibt die aktuelle Softwarearchitektur von `diffpf`. Sie soll als kompakter Kontext für Entwicklung, Dokumentation und Coding-Agenten dienen.

## Grundprinzip

`diffpf` trennt strikt zwischen:

```text
menschen-/toolnaher Eingabe
    -> kanonischer Netzbeschreibung
        -> statischer Kompilierung
            -> JAX-kompatiblem numerischem Kern
```

Der numerische Kern soll keine `pandapower`-, JSON- oder Parserlogik enthalten.

## Methodische Fixpunkte

- Spannungen werden intern komplex verwendet: `V = V_r + j V_i`.
- Der freie Solverzustand ist reell und enthält die Nicht-Slack-Busse.
- Die elektrische Physik lautet: `I = Y_bus @ V`, `S = V * conj(I)`.
- Es wird im Per-Unit-System gerechnet.
- Die Topologie ist statisch; Parameter in `NetworkParams` sind differenzierbar.
- Der Power Flow wird mit Newton-Raphson gelöst.
- Gradienten durch den gelösten Betriebspunkt werden über implizite Differentiation berechnet, nicht über vollständiges Solver-Unrolling.
- Numerische Hot Paths müssen JAX-/JIT-kompatibel bleiben.
- Keine Python-`if`-Abfragen auf JAX-Tracer-Werten im Kern.

## Zentrale Datenpipeline

```text
pandapower.to_json oder pandapower-Netzobjekt
    -> RawPandapowerNetwork / RawNetwork
    -> NetworkSpec
    -> compile_network(...)
    -> CompiledTopology + NetworkParams + PFState
    -> Newton / implicit solver
    -> Observables / Experimente
```

## Schichten

### `src/diffpf/io/`

Zuständig für Eingabe, Validierung und externe Formate.

Typische Aufgaben:

- JSON-Netze laden,
- Raw-Dataclasses validieren,
- physikalische Größen in Per-Unit umrechnen,
- externe Bus-IDs auf interne Indizes mappen,
- `pandapower`-Netze in `NetworkSpec` überführen,
- Switch-Vorverarbeitung und Bus-Fusion.

Wichtige Bestandteile:

- `reader.py`: Raw-Dataclasses und JSON-Reader.
- `parser.py`: RawNetwork -> NetworkSpec / kompilierte Strukturen.
- `pandapower_adapter.py`: `from_pandapower(net)` und `load_pandapower_json(path)`.
- `topology_utils.py`: Bus-Fusion über Union-Find.

Regel: `pandapower` darf hier verwendet werden. `core/`, `solver/` und `compile/` dürfen nicht aus `io/` importieren.

### `src/diffpf/core/`

Reiner numerischer Kern.

Wichtige Dateien:

- `types.py`: zentrale Datentypen.
- `units.py`: Per-Unit-Basisgrößen.
- `ybus.py`: Stamping von Leitungen, Transformatoren und Shunts.
- `residuals.py`: Spannungsabbildung, Leistungsberechnung, Residuen.
- `observables.py`: Auswertung gelöster Betriebspunkte.

Wichtige Typen:

- `BusSpec`
- `LineSpec`
- `TrafoSpec`
- `ShuntSpec`
- `NetworkSpec`
- `CompiledTopology`
- `NetworkParams`
- `PFState`

### `src/diffpf/compile/`

Zuständig für die statische Kompilierung von `NetworkSpec` nach JAX-kompatiblen Arrays.

Regeln:

- Keine `pandapower`-Abhängigkeit.
- Keine Parserlogik.
- Topologie gehört in `CompiledTopology`.
- Differenzierbare Parameter gehören in `NetworkParams`.

### `src/diffpf/solver/`

Zuständig für Newton-Solve und implizite Differentiation.

Wichtige Bestandteile:

- `newton.py`: gedämpfter Newton-Raphson-Solver.
- `implicit.py`: implizite Differentiation über `jax.lax.custom_root`.

Der implizite Solver verwendet im Forward-Pass weiterhin den bestehenden Newton-Solver. Es darf keine zweite physikalische Formulierung entstehen.

### `src/diffpf/validation/`

Zuständig für Referenzvergleiche und numerische Diagnostik.

Typische Aufgaben:

- Finite Differences,
- Vergleich gegen `pandapower`,
- AD-vs-FD-Gradientenvalidierung.

`pandapower` darf hier verwendet werden.

### `src/diffpf/models/`

Zuständig für vorgelagerte JAX-kompatible Modellbausteine.

Aktuell vorbereitet:

- `pv.py`: PV-P/Q-Kopplungsinterface für `example_simple()`.

Festgelegter Kopplungspunkt:

```text
Bus:        "MV Bus 2"
Element:    sgen "static generator"
P_base:     2.0 MW
Q_base:    -0.5 MVAr
Q/P:       -0.25
```

### `experiments/`

Ein Skript pro wissenschaftlichem Experiment oder Check.

Aktuell relevante Skripte:

- `exp01_validate_pandapower.py`: ursprüngliche 3-Bus-Validierung.
- `exp01_validate_example_simple.py`: Vorwärtsvalidierung `example_simple()`.
- `exp02_validate_gradients.py`: ursprüngliche 3-Bus-Gradientenvalidierung.
- `exp02_validate_gradients_example_simple.py`: Gradientenvalidierung `example_simple()`.
- `check_pv_coupling_baseline.py`: Baseline-Reproduktionscheck für PV-Kopplung.

Ab Experiment 3 sollen neue Experimente ausschließlich auf `example_simple()` aufbauen.

### `tests/`

Absicherung von Kern, I/O, Compiler, Solver, Observables, Validierung und Experiment-Artefakten.

Neue Modellbausteine müssen Tests erhalten. Schwere numerische Integrationstests können markiert werden, Schema- und Unit-Tests sollen leichtgewichtig bleiben.

## Unterstützte `pandapower`-Elemente

Aktuell unterstützt:

- `bus`, nach Switch-Fusion,
- genau ein `ext_grid` als Slack,
- `load` als negative P/Q-Injektion mit `scaling`,
- `sgen` als positive feste P/Q-Injektion mit `scaling`,
- `gen` als feste P-Einspeisung im scope-matched Modell, ohne aktive Spannungsregelung,
- `line` als Pi-Modell,
- `trafo` als 2-Wicklungs-Pi-Modell mit Tap/Shift,
- `shunt` als Y-Bus-Diagonaladmittanz,
- geschlossene Bus-Bus-Switches als Bus-Fusion,
- offene Line-Switches als deaktivierte Leitung,
- offene Trafo-Switches als deaktivierter Trafo, sofern im Adapter vorgesehen.

Nicht unterstützt:

- `trafo3w`,
- `ward` / `xward`,
- `impedance`,
- `dcline`,
- Controller,
- vollständige PV-Bus-Spannungsregelung mit Q-Ergebnis,
- Q-Limits und PV↔PQ-Umschaltung.

## Designregeln

1. Keine `pandapower`-Logik in `core/`, `solver/` oder `compile/`.
2. Keine neuen globalen Zustände im numerischen Kern.
3. Differenzierbare Größen müssen in `NetworkParams` liegen.
4. Topologieänderungen erfolgen vor dem Compile-Schritt.
5. Experimente dürfen Adapterlogik nutzen, sollen aber den Kern nicht verändern.
6. Neue Artefakte sollen tidy CSV/JSON exportieren.
7. Bekannte Modellvereinfachungen müssen in `metadata.json`, README und Changelog dokumentiert werden.
8. Jede Erweiterung braucht Tests und einen Changelog-Eintrag.
