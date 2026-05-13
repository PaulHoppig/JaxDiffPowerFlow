"""
diffpf.io – JSON loading and parsing for network definitions.

The two-layer design keeps raw Python (reader) strictly separated
from JAX array construction (parser).

  reader.py             : JSON file → RawNetwork  (plain Python, no JAX)
  parser.py             : RawNetwork → (CompiledTopology, NetworkParams, PFState)
  pandapower_adapter.py : pandapower net → NetworkSpec  (pandapower import here only)
  topology_utils.py     : Bus-Fusion-Hilfsfunktionen
"""

from .parser import line_to_pu, load_network, parse
from .reader import RawNetwork, RawShunt, RawTrafo, load_json

# pandapower_adapter is imported lazily to avoid requiring pandapower at
# import time (it may not be installed in all environments).
def from_pandapower(net):
    """Wandelt ein pandapower-Netzobjekt in eine NetworkSpec um."""
    from .pandapower_adapter import from_pandapower as _from_pandapower
    return _from_pandapower(net)


def load_pandapower_json(path):
    """Lädt ein pandapower-JSON und gibt NetworkSpec zurück."""
    from .pandapower_adapter import load_pandapower_json as _load_pp_json
    return _load_pp_json(path)


__all__ = [
    # JSON-Format
    "load_json",
    "load_network",
    "parse",
    "line_to_pu",
    "RawNetwork",
    "RawTrafo",
    "RawShunt",
    # pandapower-Adapter
    "from_pandapower",
    "load_pandapower_json",
]
