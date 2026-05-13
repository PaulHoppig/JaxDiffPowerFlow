# Modellierungsannahmen

## Netzmodell

`diffpf` modelliert stationären, symmetrischen AC-Power-Flow. Es werden komplexe Spannungen und Ströme verwendet:

```text
I = Y_bus @ V
S = V * conj(I)
```

Der Solverzustand ist reell parametrisiert. Standardmäßig enthält er Real- und Imaginärteile der Nicht-Slack-Busspannungen.

## Per-Unit-System

Alle elektrischen Gleichungen werden im Per-Unit-System ausgewertet.

Grundannahmen:

- `S_base` ist netzweit definiert.
- `V_base` muss bei mehrstufigen Netzen je Bus bzw. Spannungsebene korrekt behandelt werden.
- `Z_base = V_base^2 / S_base`.
- `Y_base = 1 / Z_base`.
- Leitungs-, Trafo- und Shuntparameter werden vor dem JAX-Kern in p.u. umgerechnet.

Bei `pandapower`-Netzen ist die Spannungsebene pro Bus (`vn_kv`) entscheidend. Die p.u.-Umrechnung darf nicht implizit von einer einzigen Spannungsebene ausgehen.

## Bus-Typen

### Slack-Bus

Der Slack-Bus hat feste Spannung nach Betrag und Winkel. Er besitzt keine freie Residualgleichung. Seine P/Q-Leistung ergibt sich aus der Lösung.

### PQ-Bus

Für PQ-Busse sind Wirkleistung und Blindleistung vorgegeben:

```text
r_P = P_spec - P_calc
r_Q = Q_spec - Q_calc
```

Lasten werden als negative Einspeisung, Erzeuger als positive Einspeisung aggregiert.

### PV-Bus im Lastfluss-Sinn

Ein idealisierter PV-Bus gibt P und Spannungsbetrag vor. Q ergibt sich aus der Lösung:

```text
r_P = P_spec - P_calc
r_V = V_set_pu^2 - |V|^2
```

Der mathematische Kern ist auf diese Form vorbereitet. Im aktuellen `pandapower`-Validierungsumfang wird ein `gen` jedoch nicht als vollständig spannungsregelnder PV-Bus mit Q-Ergebnis behandelt.

## PV-Anlage vs. PV-Bus

Eine Photovoltaikanlage ist nicht automatisch ein PV-Bus im Lastfluss-Sinn.

Für die Upstream-Kopplung wird die PV-Anlage als vorgelagertes P/Q-Einspeisemodell behandelt:

```text
Einstrahlung, Temperatur -> PV-Modell -> P_pv, Q_pv -> PQ-Injektion am Bus
```

Der Kopplungsbus `"MV Bus 2"` bleibt ein PQ-Bus. Die PV-Anlage regelt im aktuellen Scope nicht aktiv die Spannung.

## `load`, `sgen`, `gen`

### `load`

`pandapower.load` wird als negative P/Q-Injektion übernommen. `scaling` wird berücksichtigt.

```text
P_spec += -p_mw * scaling
Q_spec += -q_mvar * scaling
```

### `sgen`

`pandapower.sgen` wird als positive feste P/Q-Injektion übernommen. `scaling` wird berücksichtigt.

```text
P_spec += p_mw * scaling
Q_spec += q_mvar * scaling
```

### `gen`

Im `scope_matched`-Vergleich wird `gen` als feste P-Einspeisung ohne aktive Spannungsregelung behandelt:

```text
P_spec += p_mw
Q_spec += 0
```

`vm_pu`, `min_q_mvar` und `max_q_mvar` werden nicht als aktive Regelung ausgewertet. Dadurch sind Abweichungen zum originalen `pandapower`-PV-Bus-Modell erwartbar.

## Leitungen

Leitungen werden als Pi-Ersatzschaltbild modelliert:

- Serienadmittanz aus R und X,
- hälftige Queradmittanz an beiden Enden,
- physikalische Parameter werden vorab in p.u. umgerechnet.

`g_us_per_km` wird aktuell nicht detailliert modelliert, sofern nicht explizit unterstützt.

## Transformatoren

Unterstützt werden 2-Wicklungs-Transformatoren als Pi-Modell mit:

- Serienimpedanz,
- optionaler Magnetisierung,
- Tap-Verhältnis,
- Phasenverschiebung.

Der komplexe Tap kann konzeptionell als

```text
a = tap_ratio * exp(j * shift_rad)
```

verstanden werden.

Für `example_simple()` ist die 150°-Phasenverschiebung besonders relevant. Flat Start kann dabei divergieren; Experimente nutzen deshalb eine trafo-shift-aware Initialisierung bzw. `pandapower init="dc"` für Referenzläufe.

## Shunts

Shunts werden als konstante Admittanz auf die Y-Bus-Diagonale gestempelt:

```text
Y[bus, bus] += g_pu + j*b_pu
```

Es gibt keine Shunt-Regelung.

## Switches und Topologie

Topologieänderungen erfolgen vor dem Compile-Schritt.

- Geschlossene Bus-Bus-Switches werden als Bus-Fusion behandelt.
- Offene Line-Switches deaktivieren vereinfacht die gesamte Leitung.
- Offene Trafo-Switches deaktivieren vereinfacht den gesamten Trafo, sofern im Adapter umgesetzt.
- Detaillierte offene-Leitungsende-Modelle werden nicht unterstützt.

Die Topologie ist während eines JAX-Solve- oder Gradientenlaufs statisch.

## Differenzierbarkeit

Differenzierbar sind kontinuierliche Parameter in `NetworkParams`, z. B.:

- P/Q-Injektionen,
- Last- und Einspeiseskalierungen,
- Shuntparameter,
- ausgewählte Leitungs- und Trafoparameter,
- Upstream-Modellparameter.

Nicht differenzierbar im aktuellen Scope sind diskrete Strukturentscheidungen wie:

- Switch-Zustände,
- Bus-Fusion,
- Element aktiv/inaktiv,
- PV↔PQ-Umschaltung,
- Controllerlogik.

## Hauptkopplung für PV-Upstream

Für `example_simple()` gilt:

```text
Bus:               "MV Bus 2"
Ersetztes Element: sgen "static generator"
P_ref:             2.0 MW
Q_ref:            -0.5 MVAr
Q/P:              -0.25
```

Bei Referenzbedingungen soll das PV-Modell die ursprüngliche `sgen`-Einspeisung reproduzieren:

```text
G = 1000 W/m²
T_cell = 25 °C
P_pv = 2.0 MW
Q_pv = -0.5 MVAr
```
