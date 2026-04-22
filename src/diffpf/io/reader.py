"""
Raw JSON loading and schema validation.

Diese Schicht verwendet ausschließlich reines Python – keine JAX-Arrays.
Ihre Aufgabe ist es, die Netz-JSON-Datei zu lesen, sie in typisierte
Python-Dataclasses (``RawNetwork``) umzuwandeln und bei strukturellen
oder semantischen Verletzungen ``ValueError`` auszulösen, bevor
JAX-Code berührt wird.

JSON-Schema-Übersicht
---------------------
::

    {
      "meta":  { "name": "<str>" },
      "base":  {
        "s_mva": <float>,          -- Scheinleistungsbasis [MVA]  (> 0)
        "v_kv":  <float>,          -- Spannungsbasis Leiter-Leiter [kV]  (> 0)
        "f_hz":  <float>           -- Netzfrequenz [Hz]  (> 0)
                                      Pflicht wenn c_nf_per_km verwendet wird;
                                      sonst optional.
      },
      "buses": [
        {
          "id":        <int>,      -- eindeutig, bevorzugt 0-basiert
          "name":      "<str>",
          "type":      "slack"|"pq"|"pv",
          "p_mw":      <float>,    -- Netto-Einspeisung [MW], Generatorvorzeichen
                                      (positiv = Einspeisung ins Netz)
          "q_mvar":    <float>,    -- Netto-Einspeisung [MVAR], gleiches Vorzeichen
          "v_mag_pu":  <float>,    -- Spannungsbetrag-Sollwert für Slack/PV  (Vorgabe 1.0)
          "v_ang_deg": <float>     -- Spannungswinkel-Referenz für Slack [°]  (Vorgabe 0.0)
        }, ...
      ],
      "lines": [
        {
          "id":       <int>,       -- eindeutig
          "name":     "<str>",     -- optionale Bezeichnung
          "from_bus": <int>,       -- referenziert Bus-"id"
          "to_bus":   <int>,       -- referenziert Bus-"id"

          -- Form A: direkte Gesamtparameter der Leitung  (Ohm / Siemens)
          "r_ohm":     <float>,    -- Serienresistanz  [Ω]  (>= 0)
          "x_ohm":     <float>,    -- Serienreaktanz   [Ω]
          "b_shunt_s": <float>,    -- GESAMT-Leitungsladesuszeptanz [S]
                                      optional, Vorgabe 0.0

          -- ODER  Form B: Beläge × Länge
          "length_km":        <float>,  -- Leitungslänge [km]  (> 0)
          "r_ohm_per_km":     <float>,  -- Widerstandsbelag  [Ω/km]  (>= 0)
          "x_ohm_per_km":     <float>,  -- Reaktanzbelag     [Ω/km]
          "b_shunt_s_per_km": <float>,  -- Suszeptanzbelag [S/km]  optional
          "c_nf_per_km":      <float>,  -- Kapazitätsbelag [nF/km] optional
                                           b = 2π·f·C; benötigt base.f_hz
          -- Hinweis: b_shunt_s_per_km und c_nf_per_km schließen sich aus.
          -- Form A und Form B schließen sich aus.
        }, ...
      ]
    }
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Raw-Schema-Dataclasses  (reines Python, kein JAX)
# ---------------------------------------------------------------------------


@dataclass
class RawBase:
    s_mva: float
    v_kv: float
    f_hz: float | None = None   # Pflicht bei c_nf_per_km; sonst optional


@dataclass
class RawBus:
    id: int
    name: str
    type: str                    # "slack" | "pq" | "pv"
    p_mw: float = 0.0            # Netto-Einspeisung [MW], Generatorvorzeichen
    q_mvar: float = 0.0          # Netto-Einspeisung [MVAR], Generatorvorzeichen
    v_mag_pu: float = 1.0        # Spannungsbetrag-Sollwert für Slack / PV
    v_ang_deg: float = 0.0       # Spannungswinkel-Referenz für Slack [°]


@dataclass
class RawLine:
    # Topologie
    id: int
    from_bus: int
    to_bus: int
    name: str = field(default="")

    # --- Form A: direkte Gesamtparameter ---
    r_ohm: float | None = None          # Serienresistanz [Ω]
    x_ohm: float | None = None          # Serienreaktanz  [Ω]
    b_shunt_s: float | None = None      # Gesamt-Ladesuszeptanz [S], Vorgabe 0.0

    # --- Form B: Beläge + Länge ---
    length_km: float | None = None
    r_ohm_per_km: float | None = None
    x_ohm_per_km: float | None = None
    b_shunt_s_per_km: float | None = None   # Suszeptanzbelag [S/km]
    c_nf_per_km: float | None = None        # Kapazitätsbelag [nF/km]


@dataclass
class RawNetwork:
    name: str
    base: RawBase
    buses: list[RawBus]     # aufsteigend nach id sortiert
    lines: list[RawLine]    # aufsteigend nach id sortiert


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

_VALID_BUS_TYPES = {"slack", "pq", "pv"}

# Feldindikatoren für die Formenerkennung
_FORM_A_INDICATORS: frozenset[str] = frozenset({"r_ohm", "x_ohm"})
_FORM_B_INDICATORS: frozenset[str] = frozenset(
    {"length_km", "r_ohm_per_km", "x_ohm_per_km"}
)


def _dataclass_from_dict(cls, data: dict):
    """Konstruiert eine Dataclass aus einem Dict; unbekannte Schlüssel werden ignoriert."""
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


def _detect_line_form(raw: RawLine) -> str:
    """
    Erkennt die Eingabeform einer Leitung anhand gesetzter Felder.

    Returns
    -------
    "A"  wenn Form-A-Felder (r_ohm, x_ohm) gesetzt sind
    "B"  wenn Form-B-Felder (length_km, r_ohm_per_km, x_ohm_per_km) gesetzt sind

    Raises
    ------
    ValueError  bei Mischformen oder fehlenden Parametern
    """
    has_a = raw.r_ohm is not None or raw.x_ohm is not None
    has_b = (
        raw.length_km is not None
        or raw.r_ohm_per_km is not None
        or raw.x_ohm_per_km is not None
    )
    if has_a and has_b:
        raise ValueError(
            f"Line {raw.id}: Mischform – Form-A-Felder (r_ohm, x_ohm) und "
            f"Form-B-Felder (r_ohm_per_km, x_ohm_per_km, length_km) "
            f"dürfen nicht gleichzeitig angegeben werden."
        )
    if not has_a and not has_b:
        raise ValueError(
            f"Line {raw.id}: Keine gültige Eingabeform gefunden. "
            f"Verwende Form A (r_ohm, x_ohm) oder "
            f"Form B (r_ohm_per_km, x_ohm_per_km, length_km)."
        )
    return "A" if has_a else "B"


def _validate_line(raw: RawLine, bus_id_set: set[int]) -> None:
    """Semantische Validierung einer einzelnen Leitung."""
    # Topologie
    if raw.from_bus not in bus_id_set:
        raise ValueError(
            f"Line {raw.id}: from_bus={raw.from_bus} nicht in der Bus-Liste."
        )
    if raw.to_bus not in bus_id_set:
        raise ValueError(
            f"Line {raw.id}: to_bus={raw.to_bus} nicht in der Bus-Liste."
        )
    if raw.from_bus == raw.to_bus:
        raise ValueError(
            f"Line {raw.id}: Selbstschleife (from_bus == to_bus == {raw.from_bus})."
        )

    form = _detect_line_form(raw)   # wirft bei Mischform / fehlendem Form

    if form == "A":
        # Vollständigkeit
        if raw.r_ohm is None or raw.x_ohm is None:
            raise ValueError(
                f"Line {raw.id} (Form A): r_ohm und x_ohm müssen beide angegeben werden."
            )
        if raw.r_ohm < 0:
            raise ValueError(
                f"Line {raw.id} (Form A): r_ohm muss >= 0 sein, got {raw.r_ohm}."
            )
        r_total, x_total = raw.r_ohm, raw.x_ohm

    else:  # Form B
        # Vollständigkeit der Pflichtfelder
        missing = [
            f
            for f, v in [
                ("length_km", raw.length_km),
                ("r_ohm_per_km", raw.r_ohm_per_km),
                ("x_ohm_per_km", raw.x_ohm_per_km),
            ]
            if v is None
        ]
        if missing:
            raise ValueError(
                f"Line {raw.id} (Form B): fehlende Pflichtfelder: {missing}."
            )
        if raw.length_km <= 0:
            raise ValueError(
                f"Line {raw.id} (Form B): length_km muss > 0 sein, got {raw.length_km}."
            )
        if raw.r_ohm_per_km < 0:
            raise ValueError(
                f"Line {raw.id} (Form B): r_ohm_per_km muss >= 0 sein, "
                f"got {raw.r_ohm_per_km}."
            )
        # Gegenseitiger Ausschluss der Shunt-Angaben
        if raw.b_shunt_s_per_km is not None and raw.c_nf_per_km is not None:
            raise ValueError(
                f"Line {raw.id} (Form B): b_shunt_s_per_km und c_nf_per_km "
                f"dürfen nicht gleichzeitig angegeben werden."
            )
        r_total = raw.r_ohm_per_km * raw.length_km
        x_total = raw.x_ohm_per_km * raw.length_km

    # Nullimpedanz-Prüfung (nach Expansion auf Gesamtwerte)
    if abs(complex(r_total, x_total)) < 1e-12:
        raise ValueError(
            f"Line {raw.id}: Nahezu-Null-Impedanz |z| < 1e-12 Ω "
            f"(r={r_total:.3g}, x={x_total:.3g}). "
            f"Unendliche Admittanz würde numerische Instabilität verursachen."
        )


def _validate(net: RawNetwork) -> None:
    """Wirft ``ValueError`` bei strukturellen oder semantischen Fehlern."""
    # Basis
    if net.base.s_mva <= 0:
        raise ValueError(f"base.s_mva muss > 0 sein, got {net.base.s_mva}.")
    if net.base.v_kv <= 0:
        raise ValueError(f"base.v_kv muss > 0 sein, got {net.base.v_kv}.")
    if net.base.f_hz is not None and net.base.f_hz <= 0:
        raise ValueError(f"base.f_hz muss > 0 sein, got {net.base.f_hz}.")

    # Busse
    bus_ids = [b.id for b in net.buses]
    if len(bus_ids) != len(set(bus_ids)):
        raise ValueError("Doppelte Bus-IDs im Netz erkannt.")

    slack_buses = [b for b in net.buses if b.type == "slack"]
    if len(slack_buses) != 1:
        raise ValueError(
            f"Genau ein Slack-Bus erforderlich, gefunden: {len(slack_buses)}."
        )

    for bus in net.buses:
        if bus.type not in _VALID_BUS_TYPES:
            raise ValueError(
                f"Bus {bus.id} ({bus.name!r}): unbekannter Typ {bus.type!r}. "
                f"Gültige Typen: {sorted(_VALID_BUS_TYPES)}"
            )

    # Leitungen
    bus_id_set = set(bus_ids)
    line_ids = [ln.id for ln in net.lines]
    if len(line_ids) != len(set(line_ids)):
        raise ValueError("Doppelte Leitungs-IDs im Netz erkannt.")

    for ln in net.lines:
        _validate_line(ln, bus_id_set)

    # f_hz wird benötigt, wenn c_nf_per_km verwendet wird
    for ln in net.lines:
        if ln.c_nf_per_km is not None and net.base.f_hz is None:
            raise ValueError(
                f"Line {ln.id}: c_nf_per_km erfordert base.f_hz, "
                f"das jedoch nicht angegeben wurde."
            )


# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------


def load_json(path: str | Path) -> RawNetwork:
    """
    Lädt und validiert eine Netz-JSON-Datei.

    Parameters
    ----------
    path : str | Path
        Pfad zur ``.json``-Datei.

    Returns
    -------
    RawNetwork
        Validiertes Rohnetz mit aufsteigend nach ``id`` sortierten
        Bussen und Leitungen.

    Raises
    ------
    FileNotFoundError
        Wenn die Datei nicht existiert.
    KeyError
        Wenn ein erforderlicher Schlüssel fehlt.
    ValueError
        Bei semantischen Validierungsfehlern.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Netz-Datei nicht gefunden: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    base = _dataclass_from_dict(RawBase, data["base"])
    buses = [_dataclass_from_dict(RawBus, b) for b in data["buses"]]
    lines = [_dataclass_from_dict(RawLine, ln) for ln in data["lines"]]

    # Kanonische Reihenfolge: aufsteigend nach id
    buses.sort(key=lambda b: b.id)
    lines.sort(key=lambda ln: ln.id)

    net = RawNetwork(
        name=data.get("meta", {}).get("name", path.stem),
        base=base,
        buses=buses,
        lines=lines,
    )
    _validate(net)
    return net
