# pandapower example_simple – Netzaufbau und Vorbereitung

Dieses Dokument beschreibt das pandapower-Beispielnetz `example_simple()` und
wie es für die Validierung des JAX-Power-Flow-Kerns vorbereitet wird.

## Netzstruktur

`pandapower.networks.example_simple()` ist ein 7-Bus-Netz mit zwei Spannungsebenen:

- **110-kV-Ebene:** 3 Busse (0, 1, 2), verbunden durch Leitung 0 (10 km)
- **20-kV-Ebene:** 4 Busse (3, 4, 5, 6), verbunden durch Leitungen 1–3
- **Kupplung:** 110/20-kV-Trafo (Bus 2 → Bus 3), 25 MVA, 150° Phasenverschiebung

Aktive Elemente:
- `ext_grid` am Bus 0 (Slack, vm=1.02 pu, va=50°)
- `gen` am Bus 5 (PV, p=6 MW, vm=1.03 pu)
- `sgen` am Bus 6 (p=2 MW, q=-0.5 Mvar)
- `load` am Bus 6 (p=2 MW, q=4 Mvar, scaling=0.6)
- `shunt` am Bus 2 (q=-0.96 Mvar kapazitiv)

Switches:
- Bus-Bus-Switches (et='b'): Bus 1↔2 und Bus 3↔4 (beide geschlossen)
- Leitungs-Switches (et='l'): LBS an Leitungen 1–3; Switch 5 (Leitung 2) ist geöffnet

## Switch-Fusion

Durch die geschlossenen Bus-Bus-Switches werden galvanische Verbindungen hergestellt:
- {1, 2} → repräsentativer Bus 1
- {3, 4} → repräsentativer Bus 3

Das reduzierte Netz hat daher 5 Busse: [0, 1, 3, 5, 6].

Switch 5 (et='l', closed=False) deaktiviert Leitung 2 → nur 3 Leitungen bleiben.

## Konkrete pandapower-I/O-Pipeline

### `from_pandapower(net)` Übersicht

Die Funktion `diffpf.io.from_pandapower(net)` konvertiert ein pandapower-Netzobjekt
in eine `NetworkSpec`, die über `compile_network()` in JAX-Arrays überführt werden kann.

```
pandapower net  →  from_pandapower(net)  →  NetworkSpec  →  compile_network()  →  JAX
```

### Mapping-Übersicht

| pandapower-Element | NetworkSpec-Entsprechung          | Bemerkung                              |
|--------------------|-----------------------------------|----------------------------------------|
| `ext_grid`         | Slack-Bus, `slack_vr/vi_pu`       | Genau eines erlaubt                    |
| `bus`              | `BusSpec` (nach Switch-Fusion)    | Bus-IDs können Lücken haben            |
| `load`             | `p_spec -= p*scaling/s_base`      | Negativer Beitrag (Verbrauch)          |
| `sgen`             | `p_spec += p*scaling/s_base`      | Positiver Beitrag (Einspeisung)        |
| `gen`              | `p_spec += p/s_base`              | PV-Bus, kein Q-Limit-Enforcement       |
| `line`             | `LineSpec` (Pi-Modell, p.u.)      | Form B: r/x per km * length            |
| `trafo`            | `TrafoSpec` (Pi-Modell, p.u.)     | Tap + Phasenverschiebung               |
| `shunt`            | `ShuntSpec` (g+jb, p.u.)         | g = P/s_base, b = -Q/s_base            |

### Switch-Handling

1. **Bus-Bus-Switches (et='b', closed=True):** Bus-Fusion via Union-Find (`topology_utils.merge_buses()`). Alle Element-Referenzen werden auf den repräsentativen Bus umgeschrieben.
2. **Leitungs-Switches (et='l', closed=False):** Leitung wird aus der Spec ausgeschlossen.
3. **Trafo-Switches (et='t', closed=False):** Trafo wird ausgeschlossen.

### Bekannte Vereinfachungen

- **PV-Busse:** Spannungsregelung wird NICHT durchgesetzt; PV-Busse werden wie PQ-Busse mit festem P behandelt. Der JAX-Kern löst alle Nicht-Slack-Busse als PQ.
- **Flat Start:** Bei Netzen mit großem Trafo-Phasenwinkel (>90°) versagt der Flat Start. Ein angepasster Startpunkt (Slack-Winkel ± Trafo-Shift) wird empfohlen.
- **Spannungsebenen:** Jede Leitung verwendet die Nennspannung ihres `from_bus` als lokale Basis. Keine netzweite einheitliche Basis.
- **Q-Grenzen:** Generatoren werden ohne Q-Limitierung modelliert.
- **Nicht unterstützt:** `trafo3w`, `xward`, `ward`, `impedance`, `dcline`.
