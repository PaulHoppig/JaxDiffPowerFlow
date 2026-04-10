# diffpf — Differenzierbarer AC-Leistungsflusskern in JAX

> Proof of Concept einer Bachelorarbeit zur End-to-End-Differenzierung
> über physikalische Modellgrenzen hinweg: von Wetterdaten und
> Anlagenparametern bis hin zu Knotenspannungen in einem AC-Stromnetz.

## Motivation

Klassische Leistungsflussrechner lösen stationäre Netzgleichungen, liefern
aber keine Gradienten über die Modellgrenze hinaus. `diffpf` baut den
Leistungsflusskern funktional in JAX nach und macht ihn Teil eines
durchgehend differenzierbaren Rechengraphen:

```
Wetter  →  PV-Modell  →  Power Flow  →  Knotenspannung
   ──────────────  ∂V/∂Wetter  ──────────────▶
```

Damit werden Aufgaben möglich, die mit klassischen Solvern nur umständlich
gehen — etwa **inverse Dimensionierung**: „Wie groß muss die PV-Fläche sein,
damit im Worst-Case-Wetterszenario kein Knoten über 250 V steigt?"

## Kernideen

- **Kartesische Koordinaten** ($e + jf$) statt Polarform — vermeidet
  Singularitäten bei kleinen Winkeln.
- **Newton-Raphson** über `jax.lax.while_loop`, jit-fähig.
- **Implicit Differentiation** über `jax.lax.custom_root`: Gradienten
  werden am Konvergenzpunkt analytisch berechnet, ohne die
  Solver-Iterationen zu unrollen.
- **Modularität** durch Pytrees und ein einheitliches Einspeisemodell-Protokoll:
  ein physikalisches PV-Modell und ein Equinox-NN sind austauschbar, ohne
  dass der Kern angefasst wird.

## Test-Case

Ein 3-Bus-Netz als PoC:

| Bus | Typ             | Rolle                |
|-----|-----------------|----------------------|
| 1   | Slack           | Umspannwerk          |
| 2   | PQ              | Wohngebiet (Last)    |
| 3   | PQ (oder PV)    | PV-Park (Einspeisung)|

Definition als reiner Daten-Pytree in `cases/three_bus_poc.py`.

## Projektstruktur (Kurzfassung)

```
src/diffpf/
  core/        # Netz-Primitive (types, ybus, residuals)
  solver/      # Newton-Raphson + custom_root
  models/      # Einspeisemodelle (pv_physical, pv_nn, load)
  pipeline/    # End-to-End-Graph (weather_to_pf, losses)
  validation/  # pandapower-Referenz, finite-diff-Checks
cases/         # Netzdefinitionen als Daten
experiments/   # Ein Script pro Experiment der Arbeit
tests/         # pytest
docs/          # theory.md + Figures
```

Eine ausführliche Begründung des Layouts und der Abhängigkeitsrichtung
steht in [`AGENTS.md`](./AGENTS.md).

## Setup

Voraussetzungen: Python ≥ 3.11, [`uv`](https://docs.astral.sh/uv/) (empfohlen)
oder `pip`.

```bash
# Repo klonen
git clone <repo-url> diffpf
cd diffpf

# Environment anlegen und Dependencies installieren
uv sync

# Oder klassisch:
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

JAX läuft im PoC auf der CPU — für ein 3-Bus-Netz ist keine GPU nötig.

## Erste Schritte

```bash
# Tests laufen lassen
pytest -q

# Ein Experiment ausführen
python experiments/exp01_validate_pandapower.py
python experiments/exp03_sensitivity_weather.py
```

## Experimente der Arbeit

| Nr. | Script                              | Ziel                                            |
|-----|-------------------------------------|-------------------------------------------------|
| 1   | `exp01_validate_pandapower.py`      | Numerische Validierung gegen pandapower         |
| 2   | `exp02_validate_gradients_fd.py`    | Gradienten gegen finite Differenzen prüfen      |
| 3   | `exp03_sensitivity_weather.py`      | Cross-Domain-Sensitivität $\partial V/\partial \text{Wetter}$ |
| 4   | `exp04_model_swap.py`               | Austausch PV-Physikmodell ↔ Equinox-NN          |
| 5   | `exp05_inverse_sizing_pv.py`        | Inverse Dimensionierung der PV-Fläche           |

## Methodische Details

Herleitung der Residuenformulierung in kartesischen Koordinaten, Struktur
der Jacobi-Matrix, Vorzeichenkonvention und die Herleitung der impliziten
Gradienten stehen in [`docs/theory.md`](./docs/theory.md).

## Arbeiten am Code

Vor dem ersten Commit:

```bash
pre-commit install
```

Konventionen (Style, JAX-Regeln, Testpflichten, Commit-Format) sind in
[`AGENTS.md`](./AGENTS.md) dokumentiert. Die Datei richtet sich primär an
Coding-Agenten, ist aber auch für menschliche Mitarbeitende die maßgebliche
Referenz.

## Status

Proof of Concept im Rahmen einer Bachelorarbeit. Nicht für den
Produktivbetrieb gedacht.

## Lizenz

Siehe [`LICENSE`](./LICENSE).