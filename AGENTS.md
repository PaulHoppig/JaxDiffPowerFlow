# AGENTS.md — Instructions for Coding Agents

Diese Datei beschreibt den Projektkontext, die Konventionen und die Guardrails für
Coding-Agenten (Claude Code, Cursor, Continue, Aider, …), die in diesem Repository
arbeiten. Menschliche Leser finden eine kürzere Einführung in der `README.md`.

---

## 1. Projektkontext

**Projekt:** `diffpf` — ein differenzierbarer AC-Leistungsflusskern in JAX.

**Ziel:** End-to-End-Differenzierung über physikalische Modellgrenzen hinweg
(Wetter → PV-Modell → Power Flow → Knotenspannung), nicht nur die Lösung
stationärer Netzgleichungen. Das Projekt ist der Proof of Concept meiner
Bachelorarbeit und arbeitet auf einem 3-Bus-Netz
(Umspannwerk, Wohngebiet, PV-Park).

**Methodischer Kern:**
- Kartesische Koordinaten ($e + jf$) statt Polarkoordinaten, um Singularitäten
  bei kleinen Spannungswinkeln zu vermeiden.
- Konsequentes p.u.-System.
- Newton-Raphson-Solver über `jax.lax.while_loop`.
- Gradienten am Konvergenzpunkt via `jax.lax.custom_root`
  (Implicit Function Theorem, **kein** Unrolling der Solver-Iterationen).

Wenn du als Agent eine dieser Entscheidungen in Frage stellen willst, tu das
bitte *explizit* in einem Kommentar oder Commit-Message — nicht durch stille
Umschreibung.

---

## 2. Tech-Stack

| Zweck                  | Tool                                      |
|------------------------|-------------------------------------------|
| Kernframework          | JAX (CPU reicht für PoC)                  |
| Neuronale Netze        | Equinox                                   |
| Optimierung            | Optax                                     |
| Referenz-Solver        | pandapower (nur in `validation/` & Tests) |
| Tests                  | pytest                                    |
| Formatierung / Linting | ruff, black                               |
| Build / Env            | uv (bevorzugt) oder hatch                 |
| Python                 | >= 3.11                                   |

**Wichtig:** pandapower ist eine *Referenz*, keine Laufzeitabhängigkeit des
Kerns. Es darf **nur** in `src/diffpf/validation/` und in `tests/` importiert
werden, niemals in `core/`, `solver/`, `models/` oder `pipeline/`.

---

## 3. Repository-Layout

```
src/diffpf/
  core/       # Netz-Primitive, rein funktional (types, units, ybus, residuals)
  solver/     # Newton-Raphson + custom_root-Kapselung
  models/     # Austauschbare Einspeisemodelle (pv_physical, pv_nn, load)
  pipeline/   # End-to-End-Rechengraph (weather_to_pf, losses)
  validation/ # pandapower-Referenz, finite-diff-Gradientencheck
  viz/        # Plot-Helfer

cases/        # Netzdefinitionen als Daten (Pytrees), z. B. three_bus_poc.py
experiments/  # Ein Script pro Arbeits-Experiment (exp01 … exp05)
notebooks/    # Exploration, nicht Teil der CI
tests/        # Spiegelt src/diffpf/
docs/         # theory.md + Figures
```

**Abhängigkeitsrichtung (darf nicht verletzt werden):**

```
core  →  solver  →  models  →  pipeline  →  experiments
                                 ↑
                              cases
validation darf alles importieren, wird aber von nichts importiert.
```

Agenten dürfen keine zyklischen Importe einführen. Wenn ein Refactoring das
nötig machen würde, stattdessen **fragen** oder einen TODO-Kommentar setzen.

---

## 4. Code-Konventionen

### 4.1 JAX-spezifisch

- **Funktional, keine Seiteneffekte.** Keine globalen Zustände, keine Klassen
  mit mutierbaren Attributen im Hot Path. Datencontainer sind `NamedTuple`s
  oder Equinox-Module, damit sie automatisch als Pytrees registriert sind.
- **Jit-Kompatibilität ist Pflicht** für alles in `core/`, `solver/`,
  `pipeline/`. Konkret: keine Python-`if`s auf Tracer-Werten, keine
  `.item()`-Aufrufe, keine shape-abhängigen Python-Schleifen über Tracer.
  Control Flow über `jax.lax.cond`, `jax.lax.scan`, `jax.lax.while_loop`.
- **Kein numpy im Hot Path.** `jax.numpy as jnp` durchgängig. `numpy` ist nur
  in `validation/`, Tests und Visualisierung erlaubt.
- **Komplexe Zahlen nur am Rand.** Innerhalb des Solvers wird mit reellen
  Vektoren `[e_1, …, e_n, f_1, …, f_n]` gearbeitet, weil `grad` komplexe
  Zahlen nur eingeschränkt unterstützt. Y_bus darf komplex sein, aber die
  Residuenfunktion gibt reelle Werte zurück.
- **Dtypes explizit.** Standard ist `float64` — dafür muss ganz oben
  `jax.config.update("jax_enable_x64", True)` gesetzt sein. Niemals
  stillschweigend auf float32 fallen.
- **`custom_root`-Signaturen beachten.** Alles Parametrische gehört in den
  `params`-Pytree, nicht in Closures, sonst differenziert JAX am falschen
  Objekt vorbei.

### 4.2 Style

- Formatierung: `black`, Zeilenlänge 100.
- Linting: `ruff` mit den Default-Regeln plus `I` (isort) und `NPY` (numpy).
- Type Hints überall in öffentlichen Funktionen. `jaxtyping` für
  Array-Shapes, wo es hilft (`Float[Array, "n n"]`).
- Docstrings im NumPy-Stil, auf **Englisch**. Bachelorarbeit selbst ist
  Deutsch, aber der Code bleibt englisch, damit er zitierfähig ist.
- Bezeichner: physikalische Größen dürfen ihre übliche Notation behalten
  (`V`, `Y_bus`, `e`, `f`, `P_inj`), auch wenn das PEP 8 gegen den Strich
  geht. In Kommentaren bitte die Einheit angeben (`# [p.u.]`).

### 4.3 Was Agenten *nicht* tun sollen

- Keine Performance-"Optimierungen" durch vorzeitiges `vmap`/`scan`-Umbauen,
  bevor der naive Code getestet und korrekt ist.
- Keine Einführung neuer Dependencies ohne Rückfrage. Insbesondere **nicht**
  PyTorch, TensorFlow, SymPy, scipy als Solver-Ersatz.
- Keine „Hilfs"-Klassenhierarchien um `NamedTuple`s herum.
- Kein stilles Umschreiben der Residuenformulierung oder der
  Koordinatenwahl — das sind methodische Entscheidungen der Arbeit.
- Keine Kommentare wie `# improved` oder `# fixed bug` ohne Angabe *was*
  verbessert/gefixt wurde.

---

## 5. Workflow-Phasen

Das Projekt wird in Phasen implementiert. Agenten sollten wissen, wo wir
gerade stehen, bevor sie größere Änderungen vorschlagen. Aktueller Stand
wird in `docs/STATUS.md` gepflegt (falls vorhanden).

1. **Phase 0** — Setup, Paketstruktur, CI.
2. **Phase 1** — `core/`: types, ybus (Stamping), residuals.
3. **Phase 2** — `solver/newton.py` mit `jax.lax.while_loop`, Validierung
   gegen pandapower.
4. **Phase 3** — `solver/implicit.py` mit `custom_root`, Gradientencheck
   via finite differences.
5. **Phase 4** — `models/pv_physical.py` + `pipeline/weather_to_pf.py`,
   Cross-Domain-Sensitivität.
6. **Phase 5** — `models/pv_nn.py` (Equinox) als Drop-in-Replacement,
   Modularitätsnachweis.
7. **Phase 6** — `pipeline/losses.py` + `experiments/exp05_inverse_sizing_pv.py`,
   inverse Dimensionierung der PV-Fläche.

Jede Phase endet mit: grünen Tests, einem lauffähigen `experiments/expNN_*.py`
und einem Eintrag in `docs/theory.md`, falls methodisch relevant.

---

## 6. Test- und Validierungsregeln

- **Jede neue Funktion in `core/` oder `solver/` braucht einen Test.**
- **Gradiententests** nutzen zentrale finite Differenzen mit `h = 1e-5` bis
  `1e-6`, Toleranz `rtol=1e-5`, `atol=1e-7`. Helfer liegt in
  `src/diffpf/validation/finite_diff.py`.
- **Referenztests** gegen pandapower: Spannungsbeträge und -winkel müssen
  auf `atol=1e-8 p.u.` übereinstimmen.
- Tests müssen deterministisch sein. PRNG-Keys werden explizit übergeben,
  nicht aus der Zeit gezogen.
- `pytest -q` muss lokal grün sein, bevor committed wird.

Agenten dürfen **keine** Tests löschen oder `@pytest.mark.skip`-Marker
setzen, um eine Suite grün zu bekommen. Wenn ein Test falsch ist, bitte
begründen und fragen.

---

## 7. Häufige Fallen (die Agenten bitte vermeiden)

Diese Liste ist aus echten Stolperfallen bei JAX-Physik-Solvern entstanden:

1. **PV-Bus-Gleichung an der richtigen Stelle.** An PV-Bussen ersetzt
   $|V|^2 = e^2 + f^2$ die Q-Gleichung, nicht die P-Gleichung. Reihenfolge
   der Residuen muss konsistent mit der Jacobi-Struktur sein.
2. **`jax.jacfwd` statt `jacrev`** im Solver — bei $2n$ Unbekannten und $2n$
   Gleichungen ist Forward-Mode günstiger und numerisch stabiler.
3. **`while_loop`-Carry muss alles enthalten**, was sich ändert. Python-Ints
   als Iterationszähler außerhalb des Carry führen zu Tracer-Fehlern.
4. **Vorzeichenkonvention Erzeuger/Verbraucher** (Erzeuger positiv) einmal
   festlegen und in `docs/theory.md` dokumentieren. Nicht pro Datei neu
   entscheiden.
5. **Basisgrößen** ($S_{base}$, $V_{base}$) sind Teil von `GridParams` und
   werden explizit übergeben, nicht als Modul-Konstanten.
6. **Keine in-place-Updates** auf JAX-Arrays. `.at[idx].set(...)` statt
   `x[idx] = ...`.

---

## 8. Commit- und PR-Konventionen

- Commits im Conventional-Commits-Stil: `feat:`, `fix:`, `test:`, `docs:`,
  `refactor:`, `chore:`.
- Ein Commit = eine logische Änderung. Keine "wip"-Sammel-Commits in `main`.
- PR-Beschreibung (auch bei Solo-Arbeit für die Historie) nennt: *was*,
  *warum*, *welche Phase*, und welche Tests betroffen sind.
- Agenten, die längere Änderungen machen, sollen am Ende eine kurze
  Zusammenfassung posten: geänderte Dateien, neue Tests, bekannte offene
  Punkte.

---

## 9. Wenn du als Agent unsicher bist

Lieber **fragen** als raten, insbesondere bei:

- Änderungen an der Residuenformulierung oder Koordinatenwahl.
- Einführung neuer Dependencies.
- Refactorings, die die Abhängigkeitsrichtung aus Abschnitt 3 berühren.
- Alles, was mit `custom_root` oder der impliziten Differenzierung zu tun
  hat — das ist der methodische Kern der Arbeit und muss exakt stimmen.

Für kleinere Dinge (Docstring ergänzen, Test hinzufügen, offensichtlichen
Typo fixen) einfach machen.

---

## 10. Kontakt / Kontext

Dieses Projekt ist eine Bachelorarbeit. Die Arbeit wird in Deutsch
geschrieben, der Code ist Englisch. Bei Fragen zur methodischen Richtung
immer den Autor konsultieren — Agenten treffen keine methodischen
Entscheidungen eigenständig.
