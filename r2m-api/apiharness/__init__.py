"""API-only harness for reproducing and extending GAM / R²-Mem.

Wires GAM's original agents to hosted APIs so the whole pipeline runs without
GPUs. See `../README.md` for the rationale and the list of deviations from the
published setup.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the upstream package importable without installing it, so R2M-official/
# stays a pristine checkout.
_UPSTREAM = Path(__file__).resolve().parents[2] / "R2M-official"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

__all__ = ["cached_generator", "datasets", "env", "pipeline", "retrievers", "scoring"]
