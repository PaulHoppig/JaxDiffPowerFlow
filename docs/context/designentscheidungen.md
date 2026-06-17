# Designentscheidungen im Softwareprojekt `diffpf`

Stand: 2026-06-15

Dieses Dokument sammelt die im Softwareprojekt `diffpf` getroffenen
Designentscheidungen und begruendet sie. Es basiert auf den Kontextdokumenten,
dem `CHANGELOG.md`, der aktuellen Implementierung unter `src/diffpf/`, den
Experiment-Skripten und den Tests.

Bei Widerspruechen zwischen Dokumenten wurde der Changelog als hoechste
Projektwahrheit behandelt. Wo noetig, wurde der aktuelle Code als zweite
Pruefebene verwendet. Das ist besonders relevant, weil einige aeltere
Uebersichtsdokumente noch vor Experiment 5c/5d oder vor der Korrektur der
Transformator-Magnetisierungsadmittanz entstanden sind.

## 1. Grundausrichtung

### 1.1 Forschungsprototyp statt vollstaendiges Netzberechnungswerkzeug

**Entscheidung:** `diffpf` ist als wissenschaftlicher Demonstrator fuer
differenzierbare stationaere AC-Netzberechnung ausgelegt, nicht als Ersatz fuer
`pandapower` oder als industrielles Netzberechnungswerkzeug.

**Umsetzung:**

- Der README, der Projektkontext und die Limitations-Dokumente grenzen den
  Scope klar ein.
- Die Experimente sind klein, kontrolliert und reproduzierbar.
- Nicht unterstuetzte `pandapower`-Elemente wie `trafo3w`, `ward`, `xward`,
  `impedance` und `dcline` werden bewusst abgewiesen.
- Controllerlogik, Q-Limits, PV-PQ-Umschaltung, dreiphasige Lastfluesse und
  vollstaendige OPF-Funktionalitaet sind nicht Teil des aktuellen Scopes.

**Begruendung:** Der wissenschaftliche Kern liegt nicht darin, klassische
Lastflusswerkzeuge zu ersetzen, sondern darin, stationaere Netzphysik als
differenzierbare Schicht in groessere Modellketten einzubetten. Ein enger Scope
reduziert Modellmehrdeutigkeit, erleichtert Validierung und verhindert, dass
diskrete Netzlogiken den differenzierbaren Rechengraphen dominieren.

### 1.2 End-to-End-Differenzierbarkeit als Leitprinzip

**Entscheidung:** Die Software wird auf Modellketten der Form

```text
Upstream-Modell -> P/Q-Injektion -> AC-Power-Flow -> elektrische Observable
```

ausgerichtet.

**Umsetzung:**

- Der elektrische Kern ist in JAX formuliert.
- Kontinuierliche, differenzierbare Groessen liegen in `NetworkParams`.
- Vorgelagerte Modelle wie analytische PV-Modelle und das NN-Surrogat liefern
  P/Q-Injektionen, die in `NetworkParams` geschrieben werden.
- Gradienten werden ueber implizite Differentiation durch den geloesten
  Betriebspunkt berechnet.

**Begruendung:** Die Arbeit soll Sensitivitaeten nicht-elektrischer
Eingangsgroessen, z. B. Einstrahlung, Temperatur, Wind oder Curtailment, auf
elektrische Zielgroessen berechenbar machen. Deshalb ist die Software nicht nur
ein Forward-Solver, sondern eine differenzierbare Rechenschicht.

## 2. Architektur- und Schichtungsentscheidungen

### 2.1 Strikte Trennung von I/O, Kompilierung, Kern, Solver und Experimenten

**Entscheidung:** Das Projekt trennt konsequent zwischen
menschen-/toolnaher Eingabe, kanonischer Netzbeschreibung, statischer
Kompilierung, numerischem Kern, Solver, Validierung, Upstream-Modellen und
Experimenten.

**Umsetzung:**

- `src/diffpf/io/`: JSON-Reader, Parser, `pandapower`-Adapter,
  Topologie-Vorverarbeitung.
- `src/diffpf/compile/`: Ueberfuehrung von `NetworkSpec` in
  `CompiledTopology` und `NetworkParams`.
- `src/diffpf/core/`: Datentypen, Per-Unit-Helfer, Y-Bus, Residuen,
  Observables.
- `src/diffpf/solver/`: Newton-Solver und implizite Differentiation.
- `src/diffpf/validation/`: Finite Differences, Referenzvergleiche und
  Gradientenchecks.
- `src/diffpf/models/`: JAX-kompatible vorgelagerte Modelle.
- `experiments/`: reproduzierbare wissenschaftliche Skripte.

**Begruendung:** Diese Trennung verhindert, dass externe Formatlogik oder
Referenzsolver in den numerischen Hot Path gelangen. Der Kern bleibt dadurch
klein, testbar, JIT-kompatibel und wiederverwendbar. Experimente koennen neue
Fragestellungen untersuchen, ohne die elektrische Grundformulierung zu
veraendern.

### 2.2 Keine `pandapower`-Abhaengigkeit im numerischen Kern

**Entscheidung:** `pandapower` darf im I/O-, Validierungs-, Experiment- und
Testkontext verwendet werden, aber nicht in `core/`, `compile/` oder
`solver/`.

**Umsetzung:**

- Der Adapter `src/diffpf/io/pandapower_adapter.py` erzeugt eine neutrale
  `NetworkSpec`.
- Danach arbeitet der Kern nur noch mit JAX-kompatiblen Strukturen.
- `core/`, `compile/` und `solver/` enthalten keine `pandapower`-Imports.

**Begruendung:** `pandapower` bleibt Referenz- und Eingabewerkzeug, aber der
eigentliche Forschungsbeitrag ist ein eigenstaendiger differenzierbarer Kern.
Die Trennung verhindert versteckte Abhaengigkeiten, macht den Kern portabel und
erlaubt kontrollierte scope-matched Vergleiche.

### 2.3 Kanonische Zwischenrepraesentation `NetworkSpec`

**Entscheidung:** Externe Netze werden zuerst in eine menschenlesbare,
toolunabhaengige `NetworkSpec` ueberfuehrt, bevor sie kompiliert werden.

**Umsetzung:**

- JSON-Netze laufen ueber `RawNetwork -> NetworkSpec`.
- `pandapower`-Netze laufen ueber `from_pandapower(net) -> NetworkSpec`.
- `NetworkSpec` enthaelt Busse, Leitungen, Transformatoren, Shunts,
  P/Q-Spezifikationen und Slack-Spannung.

**Begruendung:** Eine kanonische Zwischenform entkoppelt externe Datenformate
von der JAX-Welt. Dadurch koennen JSON und `pandapower` dieselbe
Kompilierungslogik nutzen, und Modellentscheidungen werden an einer klaren
Grenze sichtbar.

### 2.4 Statische Topologie getrennt von differenzierbaren Parametern

**Entscheidung:** Topologische Informationen und kontinuierliche physikalische
Parameter werden in getrennten Pytree-Strukturen gehalten.

**Umsetzung:**

- `CompiledTopology` enthaelt Buszahlen, Slack-Index, Leitungsindizes,
  variable Busse und PQ/PV-Masken.
- `NetworkParams` enthaelt P/Q-Injektionen, Leitungsparameter,
  Trafo-Parameter, Shunts und Slack-Spannung.
- Die Dataclasses sind als JAX-Pytrees registriert; statische Felder werden als
  `meta_fields`, differenzierbare Felder als `data_fields` behandelt.

**Begruendung:** JAX/JIT benoetigt stabile Strukturen. Diskrete Topologie ist
nicht differenzierbar, kontinuierliche Parameter dagegen schon. Die Trennung
erlaubt effiziente Kompilierung und klare Gradientenpfade.

## 3. Numerischer Kern

### 3.1 JAX und 64-bit-Genauigkeit

**Entscheidung:** Der numerische Kern wird in JAX mit aktivierter
64-bit-Genauigkeit implementiert.

**Umsetzung:**

- `src/diffpf/__init__.py` setzt `jax_enable_x64 = True`.
- Arrays in Kernstrukturen werden explizit als `float64`, `complex128` oder
  `int32` angelegt.

**Begruendung:** Lastflussprobleme und Gradientenvergleiche sind numerisch
sensitiv. 64-bit-Genauigkeit reduziert Rundungsfehler, verbessert die
Vergleichbarkeit mit `pandapower` und macht AD-vs-FD-Validierungen robuster.

### 3.2 Per-Unit-System im elektrischen Kern

**Entscheidung:** Alle elektrischen Gleichungen werden intern im Per-Unit-
System ausgewertet.

**Umsetzung:**

- `BaseValues` kapselt `S_base`, `V_base`, `Z_base`, `Y_base` und
  Umrechnungen zwischen MW/MVAr/kV/Ohm/Siemens und p.u.
- Leitungs-, Trafo- und Shuntparameter werden vor dem Kern nach p.u.
  konvertiert.
- Bei `pandapower`-Netzen wird die lokale Spannungsebene der jeweiligen Busse
  beruecksichtigt.

**Begruendung:** Das Per-Unit-System skaliert elektrische Groessen
vergleichbar, stabilisiert numerische Gleichungen und entspricht der ueblichen
Modellierung in der Netzberechnung. Gerade bei mehreren Spannungsebenen waere
eine direkte SI-Formulierung fehleranfaelliger.

### 3.3 Rechteckige Spannungsdarstellung

**Entscheidung:** Der freie Solverzustand verwendet Real- und Imaginaerteile
der Nicht-Slack-Busspannungen.

**Umsetzung:**

- `PFState` speichert `vr_pu` und `vi_pu`.
- `state_to_voltage()` rekonstruiert daraus den komplexen Spannungsvektor.
- Der Slack-Bus wird aus `slack_vr_pu` und `slack_vi_pu` gesetzt.

**Begruendung:** Rechteckkoordinaten vermeiden Winkel-Singularitaeten und
lassen sich direkt mit JAX-Autodiff verarbeiten. Gleichzeitig bleibt die
physikalische Formulierung intern komplex und gut lesbar.

### 3.4 Komplexe AC-Physik als gemeinsame Grundformel

**Entscheidung:** Der Kern nutzt die Standardform

```text
I = Y_bus @ V
S = V * conj(I)
```

als zentrale elektrische Berechnung.

**Umsetzung:**

- `build_ybus()` erzeugt die komplexe Admittanzmatrix.
- `calc_power_injection()` berechnet Strom- und Leistungsinjektionen.
- `power_flow_residual()` vergleicht spezifizierte und berechnete Leistungen.

**Begruendung:** Diese Form ist kompakt, physikalisch transparent und dient als
einheitliche Grundlage fuer Forward-Solve, Observables, Validierung und
Gradienten. Dadurch gibt es keine parallelen physikalischen Formulierungen.

### 3.5 Slack-, PQ- und vorbereitete PV-Bus-Residuen

**Entscheidung:** Der Kern unterstuetzt Slack- und PQ-Busse sowie eine
idealisierte PV-Bus-Residualform. Im validierten `pandapower`-Scope wird ein
`gen` jedoch nicht als vollstaendig regelnder PV-Bus verwendet.

**Umsetzung:**

- Slack-Busse haben keine freie Residualgleichung.
- PQ-Busse verwenden `P_spec - P_calc` und `Q_spec - Q_calc`.
- PV-Busse koennen `P_spec - P_calc` und eine Spannungsbetragsgleichung
  verwenden.
- Im scope-matched `example_simple()` wird der aktive `gen` als feste
  P-Einspeisung mit `Q=0` behandelt.

**Begruendung:** Der Kern ist konzeptionell erweiterbar, aber die validierten
Experimente vermeiden noch die volle Generator-Q- und PV-PQ-Umschaltlogik.
Diese Logik ist diskret und wuerde den differenzierbaren Scope deutlich
komplizieren. Der scope-matched Modus erlaubt trotzdem strikte numerische
Vergleiche.

### 3.6 Y-Bus-Stempelung fuer Leitungen, Transformatoren und Shunts

**Entscheidung:** Netzbetriebsmittel werden ueber ein Stamping-Verfahren in
die Y-Bus-Matrix eingebracht.

**Umsetzung:**

- Leitungen werden als Pi-Modell mit Serienadmittanz und halber
  Queradmittanz an beiden Enden gestempelt.
- Zweiwicklungs-Transformatoren werden mit Serienadmittanz, Magnetisierung,
  Tap und Phasenschieber gestempelt.
- Shunts werden direkt auf die Y-Bus-Diagonale addiert.
- Seit der Changelog-Aenderung vom 2026-05-19 wird die gesamte
  Trafo-Magnetisierungsadmittanz halbiert auf HV- und LV-Seite verteilt; der
  HV-Selbstadmittanzterm wird mit `tap * conj(tap)` transformiert.

**Begruendung:** Stamping ist die natuerliche und modulare Form, um
Betriebsmittel in einer Admittanzmatrix zu aggregieren. Die Trafo-Korrektur
wurde vorgenommen, weil ein frueherer Wirkverlustoffset von ca. 14 kW im
scope-matched `example_simple()` dadurch auf wenige Watt reduziert wurde.

### 3.7 Newton-Raphson als Forward-Solver

**Entscheidung:** Der stationaere Betriebspunkt wird mit einem gedaempften
Newton-Raphson-Verfahren geloest.

**Umsetzung:**

- `solve_power_flow_result()` nutzt `jax.lax.while_loop`.
- Die Jacobi-Matrix wird mit `jax.jacfwd` aus dem Residuum gebildet.
- `NewtonResult` enthaelt Loesung, Residualnorm, Loss, Iterationszahl und
  Konvergenzflag.

**Begruendung:** Newton-Raphson ist fuer stationaere AC-Lastflussprobleme
klassisch, lokal schnell konvergent und passt gut zur Root-Problem-Sicht der
impliziten Differentiation. Die strukturierte Ergebnisrueckgabe ermoeglicht
Experimente mit sauberer Diagnostik.

### 3.8 Implizite Differentiation statt Solver-Unrolling

**Entscheidung:** Gradienten durch den geloesten Power Flow werden ueber
`jax.lax.custom_root` implizit differenziert, nicht durch Unrolling aller
Newton-Iterationen.

**Umsetzung:**

- `solve_power_flow_implicit()` nutzt im Forward-Pass den bestehenden
  Newton-Solver.
- Der Rueckwaertsmodus loest das linearisierte Root-Problem am
  Konvergenzpunkt.
- Es gibt keine zweite physikalische Residuenformulierung fuer die
  Gradienten.

**Begruendung:** Der Gradient soll den stationaeren geloesten Betriebspunkt
beschreiben, nicht die numerischen Details eines bestimmten Iterationspfades.
Implizite Differentiation ist dafuer methodisch sauberer, speicherschonender
und besser an die wissenschaftliche Fragestellung angepasst.

### 3.9 Solver-unabhaengige Observables

**Entscheidung:** Elektrische Auswertungsgroessen werden in einer eigenen
Schicht berechnet und nicht in Solver oder Experimente eingebettet.

**Umsetzung:**

- `power_flow_observables()` berechnet Spannungsbetraege, Winkel,
  Slack-Leistung, Gesamtverluste und Leitungsfluesse aus einem geloesten
  Zustand.
- Dieselben Observables koennen nach Newton- oder implizitem Solve genutzt
  werden.

**Begruendung:** Die Trennung vermeidet Duplikate und haelt die Solver auf das
Root-Problem fokussiert. Experimente koennen einheitlich auf dieselben
netzseitigen Groessen zugreifen.

## 4. I/O- und Netzimportentscheidungen

### 4.1 Semantische Validierung vor JAX-Kontakt

**Entscheidung:** JSON-Netze werden vor der JAX-Kompilierung als reine
Python-Dataclasses eingelesen und validiert.

**Umsetzung:**

- `reader.py` erzeugt `RawBase`, `RawBus`, `RawLine`, `RawTrafo`, `RawShunt`
  und `RawNetwork`.
- Es prueft eindeutige IDs, genau einen Slack-Bus, gueltige Leitungsendpunkte,
  keine Self-Loops, positive Basisgroessen, gueltige Leitungsformen und
  keine Nullimpedanzen.

**Begruendung:** Fehler in Eingabedaten sollen frueh, mit fachlich lesbaren
Fehlermeldungen und ohne JAX-Tracer-Kontext auftreten. Das erleichtert
Debugging und verhindert schwer interpretierbare numerische Folgefehler.

### 4.2 Physikalische Leitungsdaten statt p.u.-Direkteingabe

**Entscheidung:** JSON-Leitungen koennen in physikalischen Einheiten angegeben
werden, entweder als Gesamtwerte oder als Belaege mal Laenge.

**Umsetzung:**

- Form A: `r_ohm`, `x_ohm`, optional `b_shunt_s`.
- Form B: `length_km`, `r_ohm_per_km`, `x_ohm_per_km`, optional
  `b_shunt_s_per_km` oder `c_nf_per_km`.
- `c_nf_per_km` erfordert `f_hz`, weil daraus `b = 2*pi*f*C` berechnet wird.
- Mischformen werden abgewiesen.

**Begruendung:** Physikalische Eingaben sind fuer Nutzer und
`pandapower`-nahe Daten plausibler als p.u.-Direktwerte. Die strikte
Formtrennung verhindert mehrdeutige oder teilweise doppelt angegebene
Leitungsparameter.

### 4.3 Kontrollierter `pandapower`-Adapter statt Vollimport

**Entscheidung:** Der `pandapower`-Import unterstuetzt nur den fuer die
Experimente benoetigten Teilumfang und weist aktive nicht unterstuetzte
Elemente ab.

**Umsetzung:**

- Unterstuetzt sind u. a. `bus`, `ext_grid`, `load`, `sgen`, vereinfachtes
  `gen`, `line`, 2-Wicklungs-`trafo`, `shunt` und ausgewaehlte Switch-Logik.
- Nicht unterstuetzte aktive Elemente fuehren zu `ValueError`.
- `load` wird als negative P/Q-Injektion, `sgen` als positive P/Q-Injektion
  und `gen` als feste P-Einspeisung ohne Q-Regelung aggregiert.

**Begruendung:** Ein partieller, expliziter Import ist wissenschaftlich
ehrlicher als ein scheinbar vollstaendiger Import mit stillen Modellluecken.
Die Experimente koennen so klar sagen, welche `pandapower`-Semantik verglichen
wird und welche nicht.

### 4.4 Scope-matched Vergleich fuer `example_simple()`

**Entscheidung:** Fuer strikte Validierung wird `example_simple()` in einem
scope-matched Modus betrachtet, in dem der aktive `gen` als feste
P-Einspeisung mit `Q=0` modelliert wird.

**Umsetzung:**

- Experiment 1b unterscheidet zwischen `scope_matched` und
  `original_pandapower`.
- Nur `scope_matched` ist ein Gleichheitstest.
- `original_pandapower` bleibt Kontextvergleich mit echter PV-Bus-Semantik in
  `pandapower`.

**Begruendung:** `pandapower` loest Generatoren mit Spannungsregelung und
Q-Ergebnis. Diese Semantik ist im aktuellen `diffpf`-Scope nicht vollstaendig
implementiert. Der scope-matched Vergleich isoliert den validierten
Modellumfang und verhindert falsche Schlussfolgerungen.

### 4.5 Topologievorverarbeitung vor dem Compile-Schritt

**Entscheidung:** Diskrete Topologieentscheidungen werden vor der
Kompilierung getroffen.

**Umsetzung:**

- Geschlossene Bus-Bus-Switches werden per Union-Find zu Busgruppen fusioniert.
- Offene Line-Switches deaktivieren vereinfacht die gesamte Leitung.
- Offene Trafo-Switches deaktivieren vereinfacht den gesamten Trafo, sofern im
  Adapter umgesetzt.
- Die Topologie bleibt waehrend Solve und Gradientenlauf statisch.

**Begruendung:** Switch-Zustaende und Bus-Fusionen sind diskrete
Strukturentscheidungen und nicht glatt differenzierbar. Durch Vorverarbeitung
bleibt der JAX-Rechengraph kontinuierlich und stabil.

## 5. Modellierungsentscheidungen fuer PV- und Upstream-Kopplung

### 5.1 PV-Anlage als PQ-Injektion, nicht als PV-Bus

**Entscheidung:** Die Photovoltaikanlage am Bus `"MV Bus 2"` wird als
wetterabhaengige P/Q-Einspeisung modelliert, nicht als spannungsregelnder
PV-Bus.

**Umsetzung:**

- Kopplungsbus: `"MV Bus 2"`.
- Ersetztes Element: `sgen "static generator"`.
- Referenzwerte: `P = 2.0 MW`, `Q = -0.5 MVAr`.
- `Q/P = -0.25`.
- `inject_pv_at_bus()` bzw. `inject_pq_at_bus()` schreibt Modelloutputs in
  `NetworkParams`.

**Begruendung:** Eine PQ-Injektion passt zum validierten Modellscope und ist
glatt differenzierbar. Spannungsregelung, Q-Limits und PV-PQ-Umschaltung sind
diskrete bzw. regelbasierte Erweiterungen und wuerden den Kern der aktuellen
Forschungsfrage ueberlagern.

### 5.2 Explizites Q/P-Verhaeltnis statt cos(phi)-Parametrisierung

**Entscheidung:** Blindleistung wird im PV-Basismodell ueber
`Q_pv = kappa * P_pv` berechnet.

**Umsetzung:**

- Standardwert `kappa = -0.25`.
- Das analytische PV-Modell und das NN-Surrogat verwenden dieselbe Kopplung.

**Begruendung:** Das Verhaeltnis reproduziert exakt das ersetzte `sgen` aus
`example_simple()` und ist einfach differenzierbar. Eine cos(phi)-Logik waere
fachlich moeglich, aber fuer den Demonstrator nicht noetig und wuerde weitere
Vorzeichen- und Grenzfallentscheidungen einfuehren.

### 5.3 Keine Saettigung oder Clipping im analytischen PV-Basismodell

**Entscheidung:** Das analytische PV-Modell wendet keine harte oder glatte
Saettigung auf die Leistung an.

**Umsetzung:**

- `pv_power_mw()` nutzt die lineare Relation
  `P = alpha * P_ref * (G/G_ref) * (1 + gamma * (T_cell - T_ref))`.
- Operational limits werden nicht im Modell versteckt, sondern in aeusseren
  Experimenten behandelt.

**Begruendung:** Clipping wuerde Nichtglattheiten oder zusaetzliche
Modellannahmen einfuehren. Ohne Saettigung bleibt der Referenzpunkt exakt,
leicht testbar und fuer AD transparent.

### 5.4 Wettergroessen bleiben in fachlichen Einheiten

**Entscheidung:** Meteorologische Eingangsgroessen werden nicht ins elektrische
p.u.-System verschoben.

**Umsetzung:**

- `g_poa_wm2`, `t_amb_c` und `wind_ms` bleiben in W/m2, Grad Celsius und m/s.
- Erst die PV-Ausgabe wird als MW/MVAr in elektrische P/Q-Injektionen
  ueberfuehrt.

**Begruendung:** Wettergroessen sind keine elektrischen Netzgroessen. Ihre
fachlichen Einheiten machen die Modellkette interpretierbar und vermeiden eine
kuenstliche Vermischung von meteorologischer und elektrischer Skalierung.

### 5.5 Reduziertes NOCT-SAM-Zelltemperaturmodell

**Entscheidung:** Die Zelltemperatur wird ueber eine reduzierte NOCT-SAM-
Relation berechnet; in dieser Version gilt `wind_adj = wind_ms`.

**Umsetzung:**

- `cell_temperature_noct_sam()` bildet Einstrahlung, Umgebungstemperatur und
  Wind auf `T_cell` ab.
- Es gibt keine Hoehen-, Montage- oder Anlagenkorrektur.

**Begruendung:** Das Modell ist einfach, JAX-kompatibel und ausreichend, um
eine Cross-Domain-Kette Wetter -> PV -> Netz zu demonstrieren. Weitere
Anlagendetails waeren fuer die Forschungsfrage nicht entscheidend und wuerden
zusaetzliche, schwer validierbare Annahmen einfuehren.

### 5.6 Einheitliche P/Q-Schnittstelle fuer analytische und neuronale Modelle

**Entscheidung:** Alle Upstream-Modelle koppeln ueber dieselbe P/Q-Schnittstelle
an den Power-Flow-Kern.

**Umsetzung:**

- Analytisches PV-Wettermodell, NN-PV-Surrogat und direkte P/Q-Baseline liefern
  `PVInjection(p_pv_mw, q_pv_mvar)`.
- Die Kopplung in `NetworkParams` erfolgt ueber dieselben Adapterfunktionen.
- Experiment 4 dokumentiert in `coupling_summary`, dass kein Modell
  Core-Aenderungen benoetigt.

**Begruendung:** Diese Entscheidung ist der Nachweis der Modularitaet: Der
elektrische Kern muss nicht wissen, ob P/Q aus einer Formel, einem neuronalen
Modell oder einer direkten Parametrisierung stammt.

## 6. NN-Surrogatentscheidungen

### 6.1 Kleines JAX-only-MLP statt ML-Framework-Stack

**Entscheidung:** Das NN-Surrogat in Experiment 4 ist ein kleines,
direkt in JAX implementiertes MLP ohne zusaetzliche ML-Frameworks wie Flax oder
Optax.

**Umsetzung:**

- `pq_surrogate.py` definiert `MLPParams`, Initialisierung, Normalisierung und
  `mlp_apply()`.
- Das Training wird im Experiment-Skript mit JAX-Funktionen realisiert.

**Begruendung:** Die Fragestellung ist Kopplungsmodularitaet, nicht
ML-Framework-Evaluation. Ein kleiner JAX-only-Ansatz minimiert Abhaengigkeiten,
macht den Rechengraphen transparent und reicht fuer das synthetische
Distillation-Surrogat.

### 6.2 P-only-Surrogat mit fester Q/P-Kopplung

**Entscheidung:** Das NN sagt nur Wirkleistung voraus; Blindleistung folgt
deterministisch aus `Q = -0.25 * P`.

**Umsetzung:**

- `neural_pq_injection_from_weather()` gibt `P_nn` und `Q_nn = kappa * P_nn`
  zurueck.
- Trainingsziel ist die normierte Wirkleistung des analytischen PV-Modells.

**Begruendung:** Dadurch bleibt das NN klein und die Blindleistungssemantik
identisch zum analytischen Modell. Unterschiede in Experiment 4/5c lassen sich
auf die P-Approximation zurueckfuehren, nicht auf eine zweite Q-Modellierung.

### 6.3 Synthetische Distillation statt Messdatenprognose

**Entscheidung:** Das NN wird auf synthetischen Daten trainiert, die aus dem
analytischen PV-Wettermodell generiert werden.

**Umsetzung:**

- Trainingsbereiche: `g_poa_wm2` von 0 bis 1200, `t_amb_c` von -10 bis 45,
  `wind_ms` von 0.5 bis 10.
- Splitgroessen im Hauptlauf: 32768 Training, 8192 Validation, 8192 Evaluation.

**Begruendung:** Das Ziel ist zu zeigen, dass ein differenzierbares
Ersatzmodell modular in die Netzphysik gekoppelt werden kann. Messdatenqualitaet
oder PV-Prognoseguete waeren eine andere Forschungsfrage.

### 6.4 Kontrollierter Kapazitaetslauf mit `hidden_width = 16`

**Entscheidung:** Der aktuelle Hauptlauf nutzt zwei versteckte Schichten der
Breite 16 mit `tanh`-Aktivierungen.

**Umsetzung:**

- `DEFAULT_HIDDEN_WIDTH = 16`.
- `DEFAULT_HIDDEN_LAYERS = 2`.
- Die Parameterzahl steigt gegenueber Width 8 von 113 auf 353.
- Changelog und Tests sichern den Width-16-Stand ab.

**Begruendung:** Der Width-16-Lauf verbessert die dokumentierten Fehlerwerte
gegenueber dem Width-8-Referenzlauf, ohne Datensatz, Modellscope,
Wetterbereiche, Loss-Funktion oder P/Q-Kopplung zu veraendern. Damit ist die
Aenderung eine kontrollierte Kapazitaetsanpassung, keine neue Modellannahme.

### 6.5 Zweiphasiges Training mit Warm-Restart-Finetune

**Entscheidung:** Das aktuelle Experiment-4-Training nutzt eine Basisphase mit
Cosine Decay und anschliessend Warm-Restart-Finetuning.

**Umsetzung:**

- Phase A: 8000 Schritte, `8e-2 -> 1e-4`.
- Phase B: 8000 Schritte in vier Warm-Restart-Zyklen.
- Globales Best-Validation-Checkpointing entscheidet ueber den finalen
  Parameterstand.
- Da kein standalone Checkpoint persistiert, reproduzieren Folgeexperimente
  den deterministischen Trainingslauf im Prozess.

**Begruendung:** Die Strategie verbessert das Surrogat innerhalb desselben
Modellscopes. Das globale Best-Checkpointing verhindert, dass ein schlechteres
Finetuning-Ergebnis den Baseline-Stand ersetzt.

## 7. Experimentdesign

### 7.1 Der 3-Bus-Fall bleibt Regression, `example_simple()` ist Hauptdemonstrator

**Entscheidung:** Der urspruengliche 3-Bus-PoC bleibt als Minimalfall erhalten;
ab Experiment 3 ist `pandapower.networks.example_simple()` der
Hauptdemonstrator.

**Umsetzung:**

- 3-Bus-PoC bleibt in `cases/` und in Basisvalidierungen.
- Experimente 3, 4 und 5 bauen auf `example_simple()` im scope-matched Modus
  auf.

**Begruendung:** Der 3-Bus-Fall ist ideal fuer schnelle Regressionen und
Grundvalidierung. `example_simple()` bietet mit Trafo, mehreren
Spannungsebenen, Switches, Shunt, Last, Gen und SGen eine realistischere
Demonstratorstruktur fuer die Bachelorarbeits-Story.

### 7.2 Reproduzierbare Skripte statt versteckter Notebooklogik

**Entscheidung:** Jedes wissenschaftliche Experiment liegt als direkt
ausfuehrbares Python-Skript unter `experiments/`.

**Umsetzung:**

- Experiment 1 bis 5d haben eigene Skripte.
- Plot-Pipelines lesen vorhandene Artefakte und starten keine schweren
  Numeriklaeufe neu.
- Tests pruefen Importierbarkeit, Schema und Pflichtartefakte.

**Begruendung:** Skripte sind leichter versionierbar, testbar und
reproduzierbar als interaktive Notebookzustaende. Plot-only-Skripte trennen
Ergebnisberechnung von Berichtsgrafik.

### 7.3 Tidy CSV/JSON-Artefakte plus Metadaten

**Entscheidung:** Experimente schreiben strukturierte CSV- und JSON-Dateien
mit flachen, tabellarischen Zeilen.

**Umsetzung:**

- Artefakte liegen unter `experiments/results/<experiment>/`.
- Jedes Experiment schreibt soweit sinnvoll `metadata.json` und `README.md`.
- Tabellen verwenden explizite Spalten fuer Szenario, Observable,
  Eingangsparameter, Wert, Einheit, Konvergenz und Diagnostik.

**Begruendung:** Tidy-Artefakte sind fuer Tests, Plots, Thesis-Tabellen und
manuelle Pruefung geeignet. JSON sichert strukturierte Weiterverarbeitung,
CSV erleichtert Tabellenarbeit.

### 7.4 Validierung in Stufen

**Entscheidung:** Die Validierung baut vom Forward-Solve ueber Gradienten bis
zu gekoppelten Upstream-Experimenten auf.

**Umsetzung:**

- Experiment 1 validiert Vorwaertssolves gegen `pandapower`.
- Experiment 2 validiert implizite AD-Gradienten gegen zentrale Finite
  Differences.
- Experiment 3 nutzt fuer die Wetterkette nur gezielte AD-vs-FD-Spot-Checks,
  weil der Gradientenkern bereits systematisch in Experiment 2 validiert ist.
- Experiment 4 validiert Modellkopplung, Power-Flow-Observables und
  Sensitivitaetsmuster fuer mehrere Upstream-Modelle.
- Experiment 5 demonstriert eine kleine gekoppelte Optimierungsaufgabe.

**Begruendung:** Diese Stufung trennt numerische Korrektheit von
Anwendungsdemonstration. Dadurch muss nicht jedes spaetere Experiment die
gesamte AD-vs-FD-Matrix erneut reproduzieren.

### 7.5 Leichtgewichtige Tests fuer schwere Experimente

**Entscheidung:** Schwere numerische Experimente werden nicht vollstaendig in
jedem Testlauf erneut ausgefuehrt; Tests sichern stattdessen Schema,
Importierbarkeit, kleine Smoke-Laeufe und vorhandene Artefakte.

**Umsetzung:**

- Tests pruefen Pflichtspalten, Metadaten und Artefaktdateien.
- Mini-Konfigurationen nutzen kleine Datensaetze oder wenige Iterationen.
- Slow-Integrationstests koennen separat markiert werden.

**Begruendung:** Die Test-Suite bleibt alltagstauglich, ohne die
Reproduzierbarkeit der Experimente aufzugeben. Schwere Hauptlaeufe bleiben
dokumentierte Artefakt- und Changelog-Ereignisse.

## 8. Entscheidungen in den Experimenten

### 8.1 Experiment 1: Zwei Referenzmodi

**Entscheidung:** Experiment 1b unterscheidet `scope_matched` und
`original_pandapower`.

**Begruendung:** Nur der scope-matched Modus vergleicht dieselbe Modellsemantik.
Der original-pandapower-Modus zeigt Kontextabweichungen, die durch echte
PV-Bus-Regelung in `pandapower` erwartbar sind.

### 8.2 Experiment 2: AD-vs-FD mit begrenztem Parameterraum

**Entscheidung:** Experiment 2b prueft 48 Gradienten auf `example_simple()`:
3 Szenarien, 4 Eingangsparameter und 4 Observables.

**Begruendung:** Der Umfang ist gross genug, um Solver, Observables und
Parameter-Mapping ernsthaft zu pruefen, bleibt aber klein genug fuer eine
kontrollierte Schrittweitenanalyse.

### 8.3 Experiment 3: Wetter -> PV -> Netz statt rein elektrischer Parameter

**Entscheidung:** Experiment 3 erweitert den Gradientenraum auf
meteorologische Eingaben.

**Umsetzung:** 93 Forward-Solves und 1116 Sensitivitaetszeilen fuer
`g_poa_wm2`, `t_amb_c` und `wind_ms`.

**Begruendung:** Dieses Experiment zeigt den Kernnutzen des Projekts: Nicht nur
elektrische Parameter, sondern vorgelagerte physikalische Eingangsgroessen
werden bis zu Netzobservables differenzierbar.

### 8.4 Experiment 4: Modularitaet vor Modellperfektion

**Entscheidung:** Experiment 4 vergleicht analytisches PV-Wettermodell,
NN-Surrogat und direkte P/Q-Baseline ueber dieselbe Kopplung.

**Begruendung:** Die Forschungsfrage lautet, ob verschiedene Upstream-Modelle
ohne Core-Aenderung gekoppelt werden koennen. Deshalb sind
Kopplungserfolg, Konvergenz, Gradienten und Sensitivitaetsmuster wichtiger als
eine perfekte PV-Prognose.

### 8.5 Experiment 5a: Screening statt sofortiger Optimierung

**Entscheidung:** Experiment 5 beginnt mit einem Forward-Screening und waehlt
erst danach einen Demonstratorfall fuer Curtailment.

**Umsetzung:** 48 PV-Screeningfaelle, no-PV-Referenzen, Top-20-
Sensitivitaeten und ein separater realistischer 30-C-Auswahlfall.

**Begruendung:** Das Screening macht transparent, warum ein bestimmter
Betriebspunkt fuer die Optimierung gewaehlt wird. Der 30-C-Auswahlfall trennt
das fachlich plausiblere Hauptnarrativ von mathematischen Stresspunkten wie
sehr kaltem Hoch-PV-Wetter.

### 8.6 Experiment 5b: Eindimensionale Curtailment-Optimierung

**Entscheidung:** Experiment 5b optimiert nur den Curtailment-Faktor eines
einzigen Betriebspunkts.

**Umsetzung:**

- Physikalische Variable `c in [0, 1]`.
- Optimiert wird `theta` mit `c(theta) = sigmoid(theta)`.
- Zielfunktion mit Softplus-Export-Penalty, `p_export_target_mw = 6.99`,
  `p_export_limit_mw = 7.0` und kleiner Curtailment-Regularisierung.
- Lokaler Adam-Loop und 1001-Punkte-Grid-Referenz.

**Begruendung:** Die eindimensionale Aufgabe ist klein genug, um die
End-to-End-Differenzierbarkeit klar zu demonstrieren. Die Sigmoid-
Parametrisierung garantiert zulaessige Curtailment-Werte, und die Grid-Referenz
erlaubt eine einfache Plausibilitaetspruefung.

### 8.7 Experiment 5c: Gleiche Optimierung mit NN-Upstream

**Entscheidung:** Experiment 5c loest dieselbe Aufgabe wie 5b, ersetzt aber das
analytische PV-Wettermodell durch das trainierte NN-Surrogat aus Experiment 4.

**Begruendung:** Damit wird gezeigt, dass die gekoppelte Optimierung nicht an
ein bestimmtes analytisches Upstream-Modell gebunden ist. Unterschiede zu 5b
sind auf die NN-Approximation zurueckzufuehren, waehrend Kern, Ziel und
Optimierungslogik gleich bleiben.

### 8.8 Experiment 5d: Einfache quadratische Zielwertsuche

**Entscheidung:** Experiment 5d nutzt fuer denselben analytischen Fall eine
reine Zielwert-Objective:

```text
objective = ((p_export_proxy - 7.0) / p_scale_mw) ** 2
```

**Umsetzung:**

- Keine Softplus-Penalty.
- Kein 6.99-MW-Ziel.
- Keine Curtailment-Regularisierung.
- `soft_export_violation_mw` bleibt nur aus Schema-Kompatibilitaet als `NaN`.

**Begruendung:** Experiment 5d isoliert die Wirkung einer einfachen
symmetrischen Zielwertsuche. Es erklaert, warum die Loesung naeher an 7.0 MW
liegt als 5b und warum ein minimaler numerischer Ueber- oder Unterschritt kein
Widerspruch zur Methode ist.

## 9. Dokumentations- und Wahrheitsmodell

### 9.1 Changelog als hoechste Wahrheit

**Entscheidung:** Der Changelog ist die massgebliche Quelle fuer den tatsaechlich
implementierten Stand.

**Umsetzung:**

- Neue Experimente und Modellkorrekturen werden im `CHANGELOG.md` dokumentiert.
- Kontextdokumente werden schrittweise aktualisiert, koennen aber zeitweise
  aeltere Aussagen enthalten.

**Begruendung:** Bei einem iterativen Forschungsprototyp entstehen Plaene,
Statusdateien und Kontextnotizen zu unterschiedlichen Zeitpunkten. Der
Changelog bietet die chronologische, implementierungsnahe Wahrheit.

### 9.2 Bekannte Grenzen werden bewusst dokumentiert

**Entscheidung:** Modellvereinfachungen und Nicht-Ziele werden nicht versteckt,
sondern in Kontextdokumenten, Experiment-READMEs, Metadaten und Changelog
festgehalten.

**Begruendung:** Die Validitaet der Bachelorarbeit haengt davon ab, dass
Ergebnisse im richtigen Scope interpretiert werden. Transparente Grenzen sind
deshalb Teil der Methode, nicht nur ein Anhang.

## 10. Aufgeloeste oder beruecksichtigte Unstimmigkeiten

### 10.1 Experimentstatus in aelteren Uebersichten

Einige Uebersichten nennen Experiment 3/4/5 noch als geplant oder beschreiben
Experiment 5 nur bis 5b. Der Changelog dokumentiert jedoch, dass Experiment 3,
4, 5a, 5b, 5c und 5d implementiert sind. Dieses Dokument folgt dem Changelog.

### 10.2 Transformator-Verlustoffset

Aeltere Validierungsbeschreibungen erwaehnen einen systematischen ca. 14-kW-
Offset bei Transformatorverlusten, Slack-Wirkleistung und Gesamtverlusten. Der
Changelog-Eintrag vom 2026-05-19 dokumentiert die Korrektur der
Trafo-Magnetisierungsadmittanz. Der aktuelle Code in `ybus.py` und
`pandapower_adapter.py` entspricht dieser korrigierten Modellierung. Dieses
Dokument behandelt den 14-kW-Offset daher als historisch weitgehend behobenen
Befund, nicht als aktuellen Hauptfehler.

### 10.3 Experiment 5b vs. 5d Zielwerte

Experiment 5b nutzt ein 6.99-MW-Ziel mit Softplus-Sicherheitsmarge und
Curtailment-Regularisierung. Experiment 5d nutzt dagegen eine reine
quadratische Zielwertsuche auf 7.0 MW. Unterschiedliche Endwerte sind deshalb
beabsichtigt und kein Widerspruch.

## 11. Gesamteinordnung

Die Designentscheidungen folgen einem einheitlichen Muster:

1. Der elektrische Kern bleibt klein, JAX-kompatibel und frei von externem
   I/O.
2. Diskrete Strukturentscheidungen werden vor dem differenzierbaren Solve
   getroffen.
3. Kontinuierliche physikalische Parameter und Upstream-Outputs werden in
   `NetworkParams` gehalten.
4. Validierung erfolgt schrittweise: Forward-Solve, Gradienten,
   Cross-Domain-Kopplung, Modularitaet und einfache Optimierung.
5. Die PV- und NN-Modelle sind bewusst einfach, weil sie die Kopplung an den
   differenzierbaren Power-Flow-Kern demonstrieren sollen.

Damit ist `diffpf` kein universelles Netzberechnungswerkzeug, sondern ein
konsistent aufgebauter Forschungsprototyp fuer differenzierbare stationaere
Netzphysik in gekoppelten Modellketten.
