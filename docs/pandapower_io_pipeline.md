# pandapower I/O-Pipeline

## Ziel

Die pandapower-I/O-Pipeline überführt pandapower-Netzobjekte in die JAX-kompatible
`NetworkSpec`-Zwischenrepräsentation. Damit können beliebige pandapower-Netze mit dem
differenzierbaren AC-Power-Flow-Kern `diffpf` gelöst und automatisch differenziert werden.

## Pipeline-Schritte

```
pandapower net-Objekt
    │
    ▼  diffpf.io.from_pandapower(net)          [pandapower_adapter.py]
    │
NetworkSpec
  ├── buses: tuple[BusSpec, ...]
  ├── lines: tuple[LineSpec, ...]
  ├── trafos: tuple[TrafoSpec, ...]
  ├── shunts: tuple[ShuntSpec, ...]
  └── p_spec_pu, q_spec_pu, slack_vr/vi_pu
    │
    ▼  diffpf.compile.network.compile_network(spec)
    │
CompiledTopology + NetworkParams
    │
    ▼  diffpf.solver.solve_power_flow(...)
    │
PFState (konvergierte Knotenspannungen)
```

### Schritt 1: from_pandapower(net)

1. **Überprüfung** auf nicht unterstützte aktive Elemente (trafo3w, xward, ward, impedance, dcline).
2. **Switch-Verarbeitung:**
   - Bus-Bus-Switches (et='b', closed=True) → Union-Find-Fusion
   - Leitungs-Switches (et='l', closed=False) → Leitung deaktiviert
   - Trafo-Switches (et='t', closed=False) → Trafo deaktiviert
3. **Bus-Mapping:** Repräsentative Busse bestimmen, interne 0-basierte Indizes vergeben.
4. **P/Q-Aggregation** je Bus:
   - loads: `-p*scaling`, `-q*scaling`
   - sgens: `+p*scaling`, `+q*scaling`
   - gens: `+p` (kein Q-Setpoint)
5. **Leitungen** → LineSpec (Pi-Modell, Form B, p.u.)
6. **Trafos** → TrafoSpec (2-Wicklungs-Pi-Modell, tap + shift)
7. **Shunts** → ShuntSpec (g + jb)

### Schritt 2: compile_network(spec)

Überführt die Python-Datenstrukturen in JAX-Arrays. Berechnet Serienadmittanzen,
stampt Trafo-Pi-Modelle, registriert Shunts.

### Schritt 3: solve_power_flow(...)

Newton-Raphson-Löser im Rechteckkoordinatensystem. JIT-kompiliert via JAX.

## Mapping-Tabelle unterstützter Elemente

| pandapower | diffpf-Spec     | Felder                                              |
|------------|-----------------|-----------------------------------------------------|
| bus        | BusSpec         | name, is_slack                                      |
| ext_grid   | Slack-Bus       | vm_pu → slack_vr/vi_pu                              |
| load       | p/q_spec_pu     | -p_mw * scaling / s_base                           |
| sgen       | p/q_spec_pu     | +p_mw * scaling / s_base                           |
| gen        | p_spec_pu       | +p_mw / s_base; vm_pu → v_set_pu (info only)       |
| line       | LineSpec        | r/x per km, c_nf/km, length, parallel              |
| trafo      | TrafoSpec       | vk%, vkr%, pfe_kw, i0%, tap, shift_deg, parallel   |
| shunt      | ShuntSpec       | p_mw*step, q_mvar*step → g_pu, b_pu                |

## Spannungsebenen-Mapping

pandapower-Netze haben typischerweise mehrere Spannungsebenen. Jedes Element
verwendet die Nennspannung des zugehörigen Busses als lokale Spannungsbasis:

- **Leitungen:** `v_base = net.bus.loc[from_bus, 'vn_kv']`
- **Trafos:** `v_base_hv = net.bus.loc[hv_bus, 'vn_kv']`
- **Systembasis:** `s_base_mva = net.sn_mva` (Default: 1.0 MVA falls 0)

Die Per-Unit-Impedanzen werden auf diese lokale Basis bezogen. Dies entspricht
dem pandapower-Ansatz (separate p.u.-Basis pro Spannungsebene).

## Vorzeichenkonventionen

| Größe          | Vorzeichen  | Bedeutung                               |
|----------------|-------------|----------------------------------------|
| p_spec_pu      | Generatorsign (+) | Einspeisung ins Netz             |
| load p_mw      | positiv = Verbrauch → negatives p_spec |                      |
| sgen p_mw      | positiv = Einspeisung → positives p_spec |                    |
| shunt q_mvar   | positiv = induktiv (absorbiert Q) → b_pu < 0 |               |
| shunt q_mvar   | negativ = kapazitiv (erzeugt Q) → b_pu > 0   |               |

## Bekannte Grenzen

1. **Kein PV-Bus-Enforcement:** Generatoren werden als PQ-Busse mit festem P und Q=0 behandelt. Die Spannungsmagnitude ist nicht geregelt.
2. **Flat-Start-Problem:** Bei Netzen mit Transformatoren >90° Phasenverschiebung (z. B. Dy-Schaltung) versagt der Standard-Flat-Start. Ein angepasster Startpunkt ist notwendig.
3. **Keine Q-Grenzen:** Generator-Q-Grenzen werden nicht berücksichtigt.
4. **Kein dynamisches Topologie-Update:** Änderungen am pandapower-Netz erfordern einen erneuten Aufruf von `from_pandapower()`.
5. **Nicht unterstützte Elemente:** trafo3w, xward, ward, impedance, dcline.
6. **Keine g_us_per_km:** Leitungsquerleitwert (g_us_per_km) wird ignoriert.
