"""
Per-unit conversion utilities.

``BaseValues`` encapsulates the system base (S_base, V_base) and exposes
bidirectional helpers between physical and per-unit quantities.
All methods operate on plain Python floats – no JAX involved.
"""

from __future__ import annotations

import math


class BaseValues:
    """
    System base quantities for per-unit conversion.

    Parameters
    ----------
    s_mva : float
        Three-phase apparent power base in MVA.
    v_kv : float
        Line-to-line voltage base in kV.
    """

    def __init__(self, s_mva: float, v_kv: float) -> None:
        if s_mva <= 0:
            raise ValueError(f"s_mva must be positive, got {s_mva}")
        if v_kv <= 0:
            raise ValueError(f"v_kv must be positive, got {v_kv}")

        self.s_mva: float = s_mva
        self.v_kv: float = v_kv

        # Derived base quantities (SI)
        self.s_base_va: float = s_mva * 1e6          # VA
        self.v_base_v: float = v_kv * 1e3            # V (line-to-line)
        self.i_base_a: float = self.s_base_va / (math.sqrt(3) * self.v_base_v)
        self.z_base_ohm: float = self.v_base_v**2 / self.s_base_va

    # ------------------------------------------------------------------
    # Physical → per-unit
    # ------------------------------------------------------------------

    def mw_to_pu(self, p_mw: float) -> float:
        """Active power: MW → p.u."""
        return p_mw / self.s_mva

    def mvar_to_pu(self, q_mvar: float) -> float:
        """Reactive power: MVAR → p.u."""
        return q_mvar / self.s_mva

    def kv_to_pu(self, v_kv: float) -> float:
        """Voltage magnitude: kV → p.u."""
        return v_kv / self.v_kv

    def ohm_to_pu(self, z_ohm: float) -> float:
        """Impedance: Ω → p.u."""
        return z_ohm / self.z_base_ohm

    def siemens_to_pu(self, y_s: float) -> float:
        """Admittance: S → p.u."""
        return y_s * self.z_base_ohm

    # ------------------------------------------------------------------
    # Per-unit → physical
    # ------------------------------------------------------------------

    def pu_to_mw(self, p_pu: float) -> float:
        """Active power: p.u. → MW."""
        return p_pu * self.s_mva

    def pu_to_mvar(self, q_pu: float) -> float:
        """Reactive power: p.u. → MVAR."""
        return q_pu * self.s_mva

    def pu_to_kv(self, v_pu: float) -> float:
        """Voltage magnitude: p.u. → kV."""
        return v_pu * self.v_kv

    def pu_to_v(self, v_pu: float) -> float:
        """Voltage magnitude: p.u. → V."""
        return v_pu * self.v_base_v

    def pu_to_ohm(self, z_pu: float) -> float:
        """Impedance: p.u. → Ω."""
        return z_pu * self.z_base_ohm

    def __repr__(self) -> str:
        return (
            f"BaseValues(s_mva={self.s_mva}, v_kv={self.v_kv}, "
            f"z_base_ohm={self.z_base_ohm:.4f})"
        )
