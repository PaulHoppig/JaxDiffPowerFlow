"""
pandapower → NetworkSpec Adapter.

Dieses Modul darf pandapower importieren. Es konvertiert pandapower-Netzobjekte
in die netzwerkunabhängige ``NetworkSpec``-Zwischenrepräsentation, die dann
über ``compile_network()`` in JAX-Arrays überführt werden kann.

Unterstützte Elemente
---------------------
- ext_grid  → Slack-Bus (genau eines)
- bus       → BusSpec (nach Switch-Fusion)
- load      → negative P/Q-Injektion
- sgen      → positive P/Q-Injektion
- gen       → PV-Bus mit P-Einspeisung
- line      → LineSpec (Pi-Modell, Form B)
- trafo     → TrafoSpec (2-Wicklungs-Trafo)
- shunt     → ShuntSpec
- switch    → Bus-Bus-Fusion oder Leitungs-/Trafo-Deaktivierung

Nicht unterstützte Elemente (ValueError wenn aktiv vorhanden)
-------------------------------------------------------------
- trafo3w, xward, ward, impedance, dcline

Architektur-Invariante
-----------------------
Kein Import von ``core/``, ``solver/``, ``compile/`` in diesem Modul verboten.
pandapower darf NUR hier (und in validation/, experiments/, tests/) importiert werden.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from diffpf.core.types import BusSpec, LineSpec, NetworkSpec, ShuntSpec, TrafoSpec
from diffpf.io.topology_utils import merge_buses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val, default: float = 0.0) -> float:
    """Gibt ``default`` zurück wenn val NaN oder None ist."""
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    return float(val)


def _check_unsupported(net) -> None:
    """Wirft ValueError bei aktiven, nicht unterstützten Elementen."""
    unsupported = {
        "trafo3w": "Dreiwicklungs-Transformatoren",
        "xward": "Extended Ward-Äquivalente",
        "ward": "Ward-Äquivalente",
        "impedance": "Impedanz-Elemente",
        "dcline": "DC-Leitungen",
    }
    for attr, label in unsupported.items():
        table = getattr(net, attr, None)
        if table is not None and len(table) > 0:
            active = table[table.in_service == True]  # noqa: E712
            if len(active) > 0:
                raise ValueError(
                    f"from_pandapower: {label} ({attr}) werden nicht unterstützt. "
                    f"Entferne oder deaktiviere diese Elemente zuerst."
                )


def _collect_switch_info(net) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """
    Verarbeitet net.switch und gibt zurück:
    - bb_pairs: geschlossene Bus-Bus-Paare für merge_buses()
    - disabled_lines: Indizes deaktivierter Leitungen (offene Leitungs-Switches)
    - disabled_trafos: Indizes deaktivierter Trafos (offene Trafo-Switches)
    """
    bb_pairs: list[tuple[int, int]] = []
    disabled_lines: set[int] = set()
    disabled_trafos: set[int] = set()

    if len(net.switch) == 0:
        return bb_pairs, disabled_lines, disabled_trafos

    for _, sw in net.switch.iterrows():
        et = sw["et"]
        closed = bool(sw["closed"])
        element = int(sw["element"])

        if et == "b":
            bus_a = int(sw["bus"])
            bus_b = element
            if closed:
                bb_pairs.append((bus_a, bus_b))
            # offene Bus-Bus-Switches: keine Wirkung (getrennte Busse bleiben getrennt)

        elif et == "l":
            if not closed:
                disabled_lines.add(element)

        elif et == "t":
            if not closed:
                disabled_trafos.add(element)

    return bb_pairs, disabled_lines, disabled_trafos



def _compute_line_spec_with_vbase(
    row,
    bus_to_repr: dict[int, int],
    repr_to_idx: dict[int, int],
    f_hz: float,
    s_base_mva: float,
    v_base_kv: float,
) -> LineSpec:
    """Konvertiert eine Leitungszeile in LineSpec mit expliziter Spannungsbasis."""
    from_bus_id = bus_to_repr[int(row["from_bus"])]
    to_bus_id = bus_to_repr[int(row["to_bus"])]
    from_idx = repr_to_idx[from_bus_id]
    to_idx = repr_to_idx[to_bus_id]

    length_km = float(row["length_km"])
    r_ohm_per_km = float(row["r_ohm_per_km"])
    x_ohm_per_km = float(row["x_ohm_per_km"])
    c_nf_per_km = _safe_float(row.get("c_nf_per_km", 0.0), 0.0)
    parallel = int(_safe_float(row.get("parallel", 1), 1))
    if parallel < 1:
        parallel = 1

    # Bei parallel > 1: Serienimpedanz /parallel, Shuntadmittanz *parallel
    r_ohm = r_ohm_per_km * length_km / parallel
    x_ohm = x_ohm_per_km * length_km / parallel
    b_shunt_s = 2.0 * math.pi * f_hz * c_nf_per_km * 1e-9 * length_km * parallel

    # Systembasis
    z_base_ohm = (v_base_kv * 1e3) ** 2 / (s_base_mva * 1e6)
    y_base_s = 1.0 / z_base_ohm

    r_pu = r_ohm / z_base_ohm
    x_pu = x_ohm / z_base_ohm
    b_shunt_pu = b_shunt_s * z_base_ohm  # b [S] * Z_base [Ω] = b [pu]

    # Nullimpedanz-Schutz
    if abs(complex(r_pu, x_pu)) < 1e-12:
        raise ValueError(
            f"Leitung {row.name}: nahezu Null-Serienimpedanz nach p.u.-Umrechnung."
        )

    return LineSpec(
        from_bus=from_idx,
        to_bus=to_idx,
        r_pu=r_pu,
        x_pu=x_pu,
        b_shunt_pu=b_shunt_pu,
    )


def _compute_trafo_spec(
    row,
    row_idx: int,
    bus_to_repr: dict[int, int],
    repr_to_idx: dict[int, int],
    s_base_mva: float,
    v_base_hv_kv: float,
) -> TrafoSpec:
    """Konvertiert eine Trafo-Zeile in TrafoSpec (Systembasis)."""
    hv_bus_id = bus_to_repr[int(row["hv_bus"])]
    lv_bus_id = bus_to_repr[int(row["lv_bus"])]
    hv_idx = repr_to_idx[hv_bus_id]
    lv_idx = repr_to_idx[lv_bus_id]

    sn_mva = float(row["sn_mva"])
    vn_hv_kv = float(row["vn_hv_kv"])
    vn_lv_kv = float(row["vn_lv_kv"])
    vk_percent = float(row["vk_percent"])
    vkr_percent = float(row["vkr_percent"])
    pfe_kw = _safe_float(row.get("pfe_kw", 0.0), 0.0)
    i0_percent = _safe_float(row.get("i0_percent", 0.0), 0.0)
    shift_degree = _safe_float(row.get("shift_degree", 0.0), 0.0)
    parallel = int(_safe_float(row.get("parallel", 1), 1))
    if parallel < 1:
        parallel = 1

    # Tap-Verhältnis berechnen
    tap_neutral = _safe_float(row.get("tap_neutral"), 0.0)
    tap_pos = _safe_float(row.get("tap_pos"), 0.0)
    tap_step_percent = _safe_float(row.get("tap_step_percent"), 0.0)
    tap_side = row.get("tap_side", "hv")
    if pd.isna(tap_side):
        tap_side = "hv"

    tap_percent = tap_step_percent / 100.0 * (tap_pos - tap_neutral)
    tap_ratio = 1.0 + tap_percent

    # Falls tap auf LV-Seite, auf HV-Seite umrechnen:
    # tap_ratio_hv = 1/tap_ratio_lv (vereinfacht: tap_ratio_lv als Faktor am LV)
    # pandapower-Konvention: tap_side='lv' bedeutet der Steller ist auf der LV-Seite.
    # Im Trafo-Modell hier: tap_ratio ist auf HV-Seite definiert.
    # Für LV-seitige Taps: Multiplikation der LV-Seite mit tap, äquivalent zu
    # Division der HV-Seite → tap_ratio_hv_equivalent = 1/tap_ratio
    if str(tap_side).lower() == "lv" and tap_ratio != 0.0:
        tap_ratio = 1.0 / tap_ratio

    shift_rad = shift_degree * math.pi / 180.0

    # Impedanzen auf Trafo-Nennbasis (HV)
    z_base_hv_ohm = (vn_hv_kv * 1e3) ** 2 / (sn_mva * 1e6)
    z_base_sys_ohm = (v_base_hv_kv * 1e3) ** 2 / (s_base_mva * 1e6)

    z_k = vk_percent / 100.0
    r_k = vkr_percent / 100.0
    x_k = math.sqrt(max(z_k**2 - r_k**2, 0.0))

    # Absolutwerte [Ω]
    z_base_t = z_base_hv_ohm
    r_ohm = r_k * z_base_t
    x_ohm = x_k * z_base_t

    # Auf Systembasis normieren
    r_pu = r_ohm / z_base_sys_ohm
    x_pu = x_ohm / z_base_sys_ohm

    # Magnetisierung (auf Trafo-Nennbasis pu)
    i0 = i0_percent / 100.0
    p_fe_pu_trafo = pfe_kw / (sn_mva * 1000.0)
    b_mag_trafo = math.sqrt(max(i0**2 - p_fe_pu_trafo**2, 0.0))

    # Auf Systembasis: Y_sys = Y_trafo * (sn_mva / s_base)
    base_factor = sn_mva / s_base_mva
    g_mag_pu = p_fe_pu_trafo * base_factor
    b_mag_pu = b_mag_trafo * base_factor

    # Bei parallel > 1: Serienimpedanz /parallel, Shuntadmittanz *parallel
    r_pu = r_pu / parallel
    x_pu = x_pu / parallel
    g_mag_pu = g_mag_pu * parallel
    b_mag_pu = b_mag_pu * parallel

    return TrafoSpec(
        hv_bus=hv_idx,
        lv_bus=lv_idx,
        r_pu=r_pu,
        x_pu=x_pu,
        g_mag_pu=g_mag_pu,
        b_mag_pu=b_mag_pu,
        tap_ratio=tap_ratio,
        shift_rad=shift_rad,
        name=str(row.get("name", f"trafo_{row_idx}")),
    )


# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------


def from_pandapower(net) -> NetworkSpec:
    """
    Wandelt ein pandapower-Netzobjekt in eine ``NetworkSpec`` um.

    Parameters
    ----------
    net : pandapower.auxiliary.pandapowerNet
        pandapower-Netz (z. B. aus ``pandapower.networks.example_simple()``).

    Returns
    -------
    NetworkSpec
        Netz-Spezifikation, die direkt an ``compile_network()`` übergeben
        werden kann.

    Raises
    ------
    ValueError
        Bei mehreren ext_grids, nicht unterstützten aktiven Elementen, oder
        inkonsistenten Busdaten.
    """
    import pandapower as pp  # hier explizit importieren (nur io-Layer)

    # --- Prüfungen ---
    _check_unsupported(net)

    active_ext = net.ext_grid[net.ext_grid["in_service"] == True]  # noqa: E712
    if len(active_ext) == 0:
        raise ValueError("from_pandapower: Kein aktives ext_grid gefunden.")
    if len(active_ext) > 1:
        raise ValueError(
            f"from_pandapower: Genau ein ext_grid unterstützt, "
            f"gefunden: {len(active_ext)}."
        )

    # --- Systembasis ---
    s_base_mva = float(net.sn_mva) if float(net.sn_mva) > 0 else 1.0
    f_hz = float(net.f_hz) if hasattr(net, "f_hz") else 50.0

    # --- Switch-Handling ---
    all_bus_ids = list(net.bus.index)
    bb_pairs, disabled_lines, disabled_trafos = _collect_switch_info(net)
    bus_to_repr = merge_buses(all_bus_ids, bb_pairs)

    # --- Repräsentative Busse bestimmen ---
    repr_bus_ids = sorted(set(bus_to_repr.values()))

    # --- Slack- und PV-Bus-Zuordnung ---
    slack_ext = active_ext.iloc[0]
    slack_original_bus = int(slack_ext["bus"])
    slack_repr_bus = bus_to_repr[slack_original_bus]
    slack_vm_pu = float(slack_ext["vm_pu"])
    slack_va_deg = float(slack_ext["va_degree"])

    # PV-Busse: Generatoren
    pv_repr_buses: dict[int, float] = {}  # repr_bus_id → vm_pu
    for _, gen_row in net.gen.iterrows():
        if not bool(gen_row["in_service"]):
            continue
        orig = int(gen_row["bus"])
        repr_id = bus_to_repr[orig]
        vm = float(gen_row["vm_pu"])
        if repr_id in pv_repr_buses and abs(pv_repr_buses[repr_id] - vm) > 1e-9:
            raise ValueError(
                f"from_pandapower: Bus-Gruppe {repr_id} hat mehrere Generatoren "
                f"mit unterschiedlichem vm_pu ({pv_repr_buses[repr_id]:.4f} vs {vm:.4f})."
            )
        pv_repr_buses[repr_id] = vm

    # Slack kann nicht gleichzeitig PV sein (ext_grid nimmt Vorrang)
    pv_repr_buses.pop(slack_repr_bus, None)

    # --- Bus-Validierung: keine Doppel-Slacks ---
    # (Da genau ein ext_grid, ist genau ein Slack-Bus definiert.)

    # --- Bus-Index-Map: repr_bus_id → interner 0-basierter Index ---
    repr_to_idx: dict[int, int] = {b: i for i, b in enumerate(repr_bus_ids)}

    # --- BusSpec-Liste aufbauen ---
    buses_list: list[BusSpec] = []
    for repr_id in repr_bus_ids:
        is_slack = (repr_id == slack_repr_bus)
        buses_list.append(BusSpec(name=str(repr_id), is_slack=is_slack))

    n_bus = len(buses_list)

    # --- P/Q-Spezifikationen (Generatorvorzeichen: + = Einspeisung) ---
    p_spec = [0.0] * n_bus
    q_spec = [0.0] * n_bus

    # Loads: negative Injektion
    for _, row in net.load.iterrows():
        if not bool(row["in_service"]):
            continue
        orig = int(row["bus"])
        repr_id = bus_to_repr[orig]
        idx = repr_to_idx[repr_id]
        scaling = _safe_float(row.get("scaling", 1.0), 1.0)
        p_spec[idx] -= float(row["p_mw"]) * scaling / s_base_mva
        q_spec[idx] -= float(row["q_mvar"]) * scaling / s_base_mva

    # sgen: positive Injektion
    for _, row in net.sgen.iterrows():
        if not bool(row["in_service"]):
            continue
        orig = int(row["bus"])
        repr_id = bus_to_repr[orig]
        idx = repr_to_idx[repr_id]
        scaling = _safe_float(row.get("scaling", 1.0), 1.0)
        p_spec[idx] += float(row["p_mw"]) * scaling / s_base_mva
        q_spec[idx] += float(row["q_mvar"]) * scaling / s_base_mva

    # gen: P-Einspeisung (Q bleibt 0.0, kein Q-Limit-Enforcement)
    for _, row in net.gen.iterrows():
        if not bool(row["in_service"]):
            continue
        orig = int(row["bus"])
        repr_id = bus_to_repr[orig]
        if repr_id == slack_repr_bus:
            continue  # Slack übernimmt P/Q automatisch
        idx = repr_to_idx[repr_id]
        p_spec[idx] += float(row["p_mw"]) / s_base_mva

    # Slack: P/Q = 0 (wird vom Solver bestimmt)
    slack_idx = repr_to_idx[slack_repr_bus]
    p_spec[slack_idx] = 0.0
    q_spec[slack_idx] = 0.0

    # --- Leitungen ---
    lines_list: list[LineSpec] = []
    for line_idx, row in net.line.iterrows():
        if not bool(row["in_service"]):
            continue
        if line_idx in disabled_lines:
            continue

        from_bus_repr = bus_to_repr[int(row["from_bus"])]
        to_bus_repr = bus_to_repr[int(row["to_bus"])]

        # Fusionierte Leitung (from == to) überspringen
        if from_bus_repr == to_bus_repr:
            continue

        v_base_kv = float(net.bus.loc[int(row["from_bus"]), "vn_kv"])

        lines_list.append(
            _compute_line_spec_with_vbase(
                row,
                bus_to_repr=bus_to_repr,
                repr_to_idx=repr_to_idx,
                f_hz=f_hz,
                s_base_mva=s_base_mva,
                v_base_kv=v_base_kv,
            )
        )

    # --- Transformatoren ---
    trafos_list: list[TrafoSpec] = []
    for trafo_idx, row in net.trafo.iterrows():
        if not bool(row["in_service"]):
            continue
        if trafo_idx in disabled_trafos:
            continue

        hv_orig = int(row["hv_bus"])
        hv_repr = bus_to_repr[hv_orig]
        lv_repr = bus_to_repr[int(row["lv_bus"])]

        # Fusionierter Trafo (hv == lv) überspringen
        if hv_repr == lv_repr:
            continue

        v_base_hv_kv = float(net.bus.loc[hv_orig, "vn_kv"])

        trafos_list.append(
            _compute_trafo_spec(
                row=row,
                row_idx=trafo_idx,
                bus_to_repr=bus_to_repr,
                repr_to_idx=repr_to_idx,
                s_base_mva=s_base_mva,
                v_base_hv_kv=v_base_hv_kv,
            )
        )

    # --- Shunts ---
    shunts_list: list[ShuntSpec] = []
    for _, row in net.shunt.iterrows():
        if not bool(row["in_service"]):
            continue
        orig = int(row["bus"])
        repr_id = bus_to_repr[orig]
        idx = repr_to_idx[repr_id]
        step = _safe_float(row.get("step", 1.0), 1.0)
        p_eff = float(row["p_mw"]) * step
        q_eff = float(row["q_mvar"]) * step
        # pandapower: positives q_mvar = induktiv → b_pu negativ
        g_pu = p_eff / s_base_mva
        b_pu = -q_eff / s_base_mva
        shunts_list.append(
            ShuntSpec(
                bus=idx,
                g_pu=g_pu,
                b_pu=b_pu,
                name=str(row.get("name", "")),
            )
        )

    # --- Slack-Spannung ---
    slack_ang_rad = slack_va_deg * math.pi / 180.0
    slack_vr = slack_vm_pu * math.cos(slack_ang_rad)
    slack_vi = slack_vm_pu * math.sin(slack_ang_rad)

    return NetworkSpec(
        buses=tuple(buses_list),
        lines=tuple(lines_list),
        p_spec_pu=tuple(p_spec),
        q_spec_pu=tuple(q_spec),
        slack_vr_pu=slack_vr,
        slack_vi_pu=slack_vi,
        trafos=tuple(trafos_list),
        shunts=tuple(shunts_list),
    )


def load_pandapower_json(path: str | Path) -> NetworkSpec:
    """
    Lädt ein pandapower-JSON und gibt eine ``NetworkSpec`` zurück.

    Parameters
    ----------
    path : str | Path
        Pfad zur pandapower-JSON-Datei (gespeichert mit ``pp.to_json()``).

    Returns
    -------
    NetworkSpec
    """
    import pandapower as pp

    path = Path(path)
    net = pp.from_json(str(path))
    return from_pandapower(net)
