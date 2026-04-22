"""
diffpf.io – JSON loading and parsing for network definitions.

The two-layer design keeps raw Python (reader) strictly separated
from JAX array construction (parser).

  reader.py  :  JSON file → RawNetwork  (plain Python, no JAX)
  parser.py  :  RawNetwork → (CompiledTopology, NetworkParams, PFState)
"""

from .parser import line_to_pu, load_network, parse
from .reader import RawNetwork, load_json

__all__ = [
    "load_json",
    "load_network",
    "parse",
    "RawNetwork",
]
