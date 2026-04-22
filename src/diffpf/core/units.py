"""
Per-unit conversion utilities.

``BaseValues`` encapsulates the system base (S_base, V_base, f_hz) and
exposes bidirectional helpers between physical and per-unit quantities.
All methods operate on plain Python floats – no JAX involved.

Basis-Gleichungen
-----------------
  Z_base  = V_base² / S_base   [Ω]
  Y_base  = S_base / V_base²   [S]  = 1 / Z_base
  I_base  = S_base / (√3 · V_base)   [A]  (Drehstrom)
"""

from __future__ import annotations

import math


class BaseValues:
    """
    Systembasis-Größen für die Per-Unit-Umrechnung.

    Parameters
    ----------
    s_mva : float
        Dreiphasige Scheinleistungsbasis [MVA].
    v_kv : float
        Verkettete Spannungsbasis (Leiter-Leiter) [kV].
    f_hz : float | None, optional
        Nennfrequenz [Hz].  Wird benötigt, wenn Leitungsladung über
        Kapazitätsbeläge (c_nf_per_km) angegeben wird.
        ``None`` ist erlaubt, solange keine kapazitive Eingabe vorliegt.
    """

    def __init__(
        self,
        s_mva: float,
        v_kv: float,
        f_hz: float | None = None,
    ) -> None:
        if s_mva <= 0:
            raise ValueError(f"s_mva must be positive, got {s_mva}")
        if v_kv <= 0:
            raise ValueError(f"v_kv must be positive, got {v_kv}")
        if f_hz is not None and f_hz <= 0:
            raise ValueError(f"f_hz must be positive, got {f_hz}")

        self.s_mva: float = s_mva
        self.v_kv: float = v_kv
        self.f_hz: float | None = f_hz

        # Abgeleitete SI-Basisgrößen
        self.s_base_va: float = s_mva * 1e6           # [VA]
        self.v_base_v: float = v_kv * 1e3             # [V]  Leiter-Leiter
        self.z_base_ohm: float = self.v_base_v**2 / self.s_base_va   # [Ω]
        self.y_base_s: float = 1.0 / self.z_base_ohm                  # [S]
        self.i_base_a: float = (
            self.s_base_va / (math.sqrt(3) * self.v_base_v)           # [A]
        )

    # ------------------------------------------------------------------
    # Physical → per-unit
    # ------------------------------------------------------------------

    def mw_to_pu(self, p_mw: float) -> float:
        """Wirkleistung: MW → p.u."""
        return p_mw / self.s_mva

    def mvar_to_pu(self, q_mvar: float) -> float:
        """Blindleistung: MVAR → p.u."""
        return q_mvar / self.s_mva

    def kv_to_pu(self, v_kv: float) -> float:
        """Spannungsbetrag: kV → p.u."""
        return v_kv / self.v_kv

    def ohm_to_pu(self, z_ohm: float) -> float:
        """Impedanz: Ω → p.u.   (r_pu = r_ohm / Z_base)"""
        return z_ohm / self.z_base_ohm

    def siemens_to_pu(self, y_s: float) -> float:
        """Admittanz / Suszeptanz: S → p.u.   (b_pu = b_S / Y_base = b_S · Z_base)"""
        return y_s * self.z_base_ohm

    # ------------------------------------------------------------------
    # Per-unit → physical
    # ------------------------------------------------------------------

    def pu_to_mw(self, p_pu: float) -> float:
        """Wirkleistung: p.u. → MW."""
        return p_pu * self.s_mva

    def pu_to_mvar(self, q_pu: float) -> float:
        """Blindleistung: p.u. → MVAR."""
        return q_pu * self.s_mva

    def pu_to_kv(self, v_pu: float) -> float:
        """Spannungsbetrag: p.u. → kV."""
        return v_pu * self.v_kv

    def pu_to_v(self, v_pu: float) -> float:
        """Spannungsbetrag: p.u. → V."""
        return v_pu * self.v_base_v

    def pu_to_ohm(self, z_pu: float) -> float:
        """Impedanz: p.u. → Ω."""
        return z_pu * self.z_base_ohm

    def pu_to_siemens(self, y_pu: float) -> float:
        """Admittanz / Suszeptanz: p.u. → S."""
        return y_pu * self.y_base_s

    def __repr__(self) -> str:
        return (
            f"BaseValues(s_mva={self.s_mva}, v_kv={self.v_kv}, "
            f"f_hz={self.f_hz}, z_base_ohm={self.z_base_ohm:.6g})"
        )
