"""
Parser: RawNetwork → (CompiledTopology, NetworkParams, PFState).

Dieses Modul ist die einzige Brücke zwischen der Raw-Python-Welt
(Dicts, JSON-gesourcte Dataclasses) und der JAX-Array-Welt.
``core/`` importiert niemals von hier.

Verantwortlichkeiten
--------------------
1. Physikalische Leitungsparameter in kanonische Gesamtform überführen
   (Form A direkt; Form B über Länge × Belag, ggf. mit C→B-Umrechnung).
2. Physikalische Gesamtparameter mit ``BaseValues`` in Per-Unit umrechnen.
3. Bus-Index-Map aufbauen (externe IDs → 0-basierte Indizes).
4. Slack-Spannungs-Sollwert in Rechteckkoordinaten zerlegen.
5. ``compile_network()`` aufrufen und flachen Startzustand erzeugen.

Öffentliche Schnittstelle
--------------------------
``parse(raw)``         – RawNetwork → (topology, params, state)
``load_network(path)`` – Komfort-One-Liner: Pfad → dasselbe Triple

Einheitenumrechnung (Leitungen)
--------------------------------
  Z_base  = V_base² / S_base              [Ω]
  Y_base  = 1 / Z_base                    [S]

  r_pu = r_ohm / Z_base
  x_pu = x_ohm / Z_base
  b_shunt_pu = b_shunt_s / Y_base = b_shunt_s · Z_base

Form B → Gesamtparameter:
  r_ohm    = r_ohm_per_km   · length_km
  x_ohm    = x_ohm_per_km   · length_km
  b_shunt_s  (aus b_shunt_s_per_km) = b_shunt_s_per_km · length_km
  b_shunt_s  (aus c_nf_per_km)      = 2π · f_hz · c_nf_per_km · 1e-9 · length_km
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp

from diffpf.compile.network import compile_network
from diffpf.core.types import BusSpec, LineSpec, NetworkSpec, PFState, ShuntSpec, TrafoSpec
from diffpf.core.units import BaseValues
from diffpf.io.reader import RawBus, RawLine, RawNetwork, RawShunt, RawTrafo, load_json


# ---------------------------------------------------------------------------
# Kanonische physikalische Zwischenform (privat, nur im Parser)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PhysicalLine:
    """
    Kanonische physikalische Leitungsparameter nach Normierung auf Gesamtwerte.

    Alle drei Felder sind Gesamtwerte der Leitung (nicht Beläge).
    ``b_shunt_s`` ist die GESAMT-Ladesuszeptanz; der Ybus-Aufbau
    verteilt sie später gleichmäßig auf beide Enden (Pi-Modell).
    """

    r_ohm: float      # Serienresistanz [Ω]  (>= 0)
    x_ohm: float      # Serienreaktanz  [Ω]
    b_shunt_s: float  # GESAMT-Ladesuszeptanz [S]  (>= 0)


# ---------------------------------------------------------------------------
# Interne Konversions-Hilfsfunktionen
# ---------------------------------------------------------------------------


def _to_physical(raw: RawLine, f_hz: float | None) -> _PhysicalLine:
    """
    Normiert eine ``RawLine`` auf kanonische physikalische Gesamtparameter.

    Form A (r_ohm, x_ohm vorhanden):
        Direkte Übernahme; ``b_shunt_s`` wird 0.0 gesetzt wenn nicht angegeben.

    Form B (length_km, r_ohm_per_km, x_ohm_per_km vorhanden):
        Multiplikation mit length_km; Shunt-Suszeptanz entweder aus
        ``b_shunt_s_per_km`` oder ``c_nf_per_km`` (b = 2π·f·C).

    Parameters
    ----------
    raw : RawLine
        Bereits validiertes RawLine-Objekt.
    f_hz : float | None
        Netzfrequenz [Hz]. Wird nur benötigt, wenn ``c_nf_per_km`` gesetzt ist
        (Validierung im Reader sichert ab, dass f_hz dann nicht None ist).
    """
    if raw.r_ohm is not None:
        # Form A
        return _PhysicalLine(
            r_ohm=raw.r_ohm,
            x_ohm=raw.x_ohm,           # type: ignore[arg-type]  # durch Validierung gesichert
            b_shunt_s=raw.b_shunt_s if raw.b_shunt_s is not None else 0.0,
        )

    # Form B
    L = raw.length_km               # type: ignore[assignment]
    r_ohm = raw.r_ohm_per_km * L   # type: ignore[operator]
    x_ohm = raw.x_ohm_per_km * L   # type: ignore[operator]

    if raw.b_shunt_s_per_km is not None:
        b_shunt_s = raw.b_shunt_s_per_km * L
    elif raw.c_nf_per_km is not None:
        # b [S] = ω · C [F] = 2π · f [Hz] · c [nF/km] · 1e-9 [F/nF] · L [km]
        b_shunt_s = 2.0 * math.pi * f_hz * raw.c_nf_per_km * 1e-9 * L  # type: ignore[operator]
    else:
        b_shunt_s = 0.0

    return _PhysicalLine(r_ohm=r_ohm, x_ohm=x_ohm, b_shunt_s=b_shunt_s)


def _physical_to_pu(
    phys: _PhysicalLine, base: BaseValues
) -> tuple[float, float, float]:
    """
    Rechnet kanonische physikalische Leitungsparameter in Per-Unit um.

    Returns
    -------
    (r_pu, x_pu, b_shunt_pu)
    """
    r_pu = base.ohm_to_pu(phys.r_ohm)
    x_pu = base.ohm_to_pu(phys.x_ohm)
    b_shunt_pu = base.siemens_to_pu(phys.b_shunt_s)
    return r_pu, x_pu, b_shunt_pu


# ---------------------------------------------------------------------------
# Weitere Hilfsfunktionen
# ---------------------------------------------------------------------------


def _raw_trafo_to_spec(
    raw: RawTrafo,
    base: BaseValues,
    id_to_idx: dict[int, int],
) -> TrafoSpec:
    """
    Wandelt einen ``RawTrafo`` in einen ``TrafoSpec`` um (Systembasis).

    Die Kurzschlussimpedanz wird von der Trafo-Nennbasis (HV-Seite) auf die
    Systembasis umgerechnet:

      z_base_hv_ohm  = (vn_hv_kv * 1e3)² / (sn_mva * 1e6)
      z_base_sys_ohm = base.z_base_ohm          (bezogen auf v_base = vn_hv_kv)
      ratio          = z_base_hv_ohm / z_base_sys_ohm

    Für die Magnetisierung:
      i0 [pu] = i0_percent / 100  (auf Trafo-Basis)
      p_fe [pu] = pfe_kw / (sn_mva * 1000)       (auf Trafo-Basis)
      Leitwerte auf Systembasis: g_mag = p_fe * sn_mva / s_base, etc.
    """
    # Trafo-eigene Impedanzbasis (Hochspannungsseite) [Ω]
    z_base_hv_ohm = (raw.vn_hv_kv * 1e3) ** 2 / (raw.sn_mva * 1e6)
    z_base_sys_ohm = base.z_base_ohm
    ratio = z_base_hv_ohm / z_base_sys_ohm

    # Kurzschlussimpedanz auf Trafo-Nennbasis [pu]
    z_k = raw.vk_percent / 100.0
    r_k = raw.vkr_percent / 100.0
    x_k = math.sqrt(max(z_k**2 - r_k**2, 0.0))

    # Auf Systembasis umrechnen
    r_pu = r_k * ratio
    x_pu = x_k * ratio

    # Magnetisierung
    i0 = raw.i0_percent / 100.0
    p_fe = raw.pfe_kw / (raw.sn_mva * 1000.0)   # pu auf Trafo-Basis
    g_mag_trafo = p_fe
    b_mag_trafo = math.sqrt(max(i0**2 - p_fe**2, 0.0))

    # Umrechnung auf Systembasis: Leitwert Y_sys = Y_trafo * (sn_mva / s_base)
    base_factor = raw.sn_mva / base.s_mva
    g_mag_pu = g_mag_trafo * base_factor
    b_mag_pu = b_mag_trafo * base_factor

    return TrafoSpec(
        hv_bus=id_to_idx[raw.hv_bus],
        lv_bus=id_to_idx[raw.lv_bus],
        r_pu=r_pu,
        x_pu=x_pu,
        g_mag_pu=g_mag_pu,
        b_mag_pu=b_mag_pu,
        tap_ratio=raw.tap_ratio,
        shift_rad=raw.shift_rad,
        name=raw.name,
    )


def _raw_shunt_to_spec(
    raw: RawShunt,
    base: BaseValues,
    id_to_idx: dict[int, int],
) -> ShuntSpec:
    """
    Wandelt einen ``RawShunt`` in einen ``ShuntSpec`` um (Systembasis).

    pandapower-Konvention: positives q_mvar = induktiv = absorbiert Blindleistung
    → negativer Suszeptanzanteil in Y.
    Der Shunt im JSON-Format trägt P+jQ als Verbrauch (wie pandapower).

    Y = (P - jQ) / s_base   [pu]  (Nennspannung ≈ Systembasis → keine Spannungskorrektur)
    """
    g_pu = raw.p_mw / base.s_mva
    b_pu = -raw.q_mvar / base.s_mva   # induktiv (positives q) → negatives b
    return ShuntSpec(
        bus=id_to_idx[raw.bus],
        g_pu=g_pu,
        b_pu=b_pu,
        name=raw.name,
    )


def _slack_rectangular(bus: RawBus) -> tuple[float, float]:
    """Gibt (vr_pu, vi_pu) für den Slack-Bus aus Polarkoordinaten zurück."""
    ang_rad = math.radians(bus.v_ang_deg)
    return bus.v_mag_pu * math.cos(ang_rad), bus.v_mag_pu * math.sin(ang_rad)


def _build_spec(raw: RawNetwork, base: BaseValues) -> NetworkSpec:
    """
    Wandelt ein validiertes ``RawNetwork`` in ein ``NetworkSpec`` um.

    Bus-Reihenfolge entspricht der sortierten ``raw.buses``-Liste (aufst. id).
    Leitungsendpunkte werden von externen IDs auf interne 0-basierte Indizes
    umgeschlüsselt.  Alle Leitungsparameter werden in p.u. umgerechnet.
    """
    # Externe Bus-ID → interner 0-basierter Index
    id_to_idx: dict[int, int] = {
        bus.id: idx for idx, bus in enumerate(raw.buses)
    }

    buses = tuple(
        BusSpec(name=b.name, is_slack=(b.type == "slack")) for b in raw.buses
    )

    # Leitungen: physikalisch → p.u. → LineSpec
    lines_list = []
    for ln in raw.lines:
        phys = _to_physical(ln, base.f_hz)
        r_pu, x_pu, b_shunt_pu = _physical_to_pu(phys, base)
        lines_list.append(
            LineSpec(
                from_bus=id_to_idx[ln.from_bus],
                to_bus=id_to_idx[ln.to_bus],
                r_pu=r_pu,
                x_pu=x_pu,
                b_shunt_pu=b_shunt_pu,
            )
        )
    lines = tuple(lines_list)

    # Vorzeichen-Konvention: JSON verwendet Generatorvorzeichen (+ = Einspeisung).
    # BaseValues rechnet mit gleichem Vorzeichen in p.u. um.
    p_spec = tuple(base.mw_to_pu(b.p_mw) for b in raw.buses)
    q_spec = tuple(base.mvar_to_pu(b.q_mvar) for b in raw.buses)

    # Transformatoren
    trafos = tuple(
        _raw_trafo_to_spec(t, base, id_to_idx) for t in raw.trafos
    )

    # Shunts
    shunts = tuple(
        _raw_shunt_to_spec(s, base, id_to_idx) for s in raw.shunts
    )

    slack_bus = next(b for b in raw.buses if b.type == "slack")
    slack_vr, slack_vi = _slack_rectangular(slack_bus)

    return NetworkSpec(
        buses=buses,
        lines=lines,
        p_spec_pu=p_spec,
        q_spec_pu=q_spec,
        slack_vr_pu=slack_vr,
        slack_vi_pu=slack_vi,
        trafos=trafos,
        shunts=shunts,
    )


def _flat_start(n_var: int) -> PFState:
    """Flat-Start-Anfangszustand: Vr = 1, Vi = 0 für alle Nicht-Slack-Busse."""
    return PFState(
        vr_pu=jnp.ones(n_var, dtype=jnp.float64),
        vi_pu=jnp.zeros(n_var, dtype=jnp.float64),
    )


# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------


def parse(raw: RawNetwork):
    """
    Wandelt ein validiertes ``RawNetwork`` in JAX-bereite Strukturen um.

    Parameters
    ----------
    raw : RawNetwork
        Ausgabe von ``reader.load_json()``.

    Returns
    -------
    topology : CompiledTopology
        Statische Integer-Index-Arrays.
    params : NetworkParams
        Differenzierbare float64-Parameter-Arrays.
    state : PFState
        Flat-Start-Anfangsspannungszustand für Newton-Raphson.
    """
    base = BaseValues(s_mva=raw.base.s_mva, v_kv=raw.base.v_kv, f_hz=raw.base.f_hz)
    spec = _build_spec(raw, base)
    topology, params = compile_network(spec)
    state = _flat_start(topology.variable_buses.shape[0])
    return topology, params, state


def line_to_pu(raw_line: RawLine, base: BaseValues) -> tuple[float, float, float]:
    """
    Rechnet eine ``RawLine`` direkt in (r_pu, x_pu, b_shunt_pu) um.

    Öffentliche Convenience-Funktion für Module, die einzelne Leitungen
    umrechnen müssen (z. B. Referenz-Solver in ``validation/``).

    Parameters
    ----------
    raw_line : RawLine
        Validierte Leitungsdefinition (Form A oder Form B).
    base : BaseValues
        Systembasis für die Per-Unit-Umrechnung.

    Returns
    -------
    (r_pu, x_pu, b_shunt_pu) : tuple[float, float, float]
    """
    phys = _to_physical(raw_line, base.f_hz)
    return _physical_to_pu(phys, base)


def load_network(path: str | Path):
    """
    One-Shot-Loader: JSON-Pfad → (topology, params, state).

    Das ist der primäre Einstiegspunkt für Nutzer, die ein Netz von Disk laden.

    Parameters
    ----------
    path : str | Path
        Pfad zur Netz-JSON-Datei.

    Returns
    -------
    topology : CompiledTopology
    params : NetworkParams
    state : PFState  (Flat-Start)

    Beispiel
    --------
    >>> topology, params, state = load_network("cases/three_bus_poc.json")
    >>> solution, norm, loss = solve_power_flow(topology, params, state)
    """
    return parse(load_json(path))
