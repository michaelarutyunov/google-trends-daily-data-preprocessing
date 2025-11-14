"""
Stitching methods for Google Trends daily data.

Provides four stitching methods:
- BaselineStitcher: Simple monthly mean scaling (fast, less accurate)
- HierarchicalStitcher: Constrained optimization with monthly + weekly targets (recommended)
- HierarchicalDOWStitcher: Hierarchical + day-of-week pattern correction (best for weekly patterns)
- SmoothAlphaStitcher: Hierarchical + smoothness penalty on alphas (recommended for production)
"""

from .base import StitchingEngine, StitchingResult
from .baseline import BaselineStitcher
from .hierarchical import HierarchicalStitcher
from .hierarchical_dow import HierarchicalDOWStitcher
from .smooth_alpha import SmoothAlphaStitcher

__all__ = [
    "StitchingEngine",
    "StitchingResult",
    "BaselineStitcher",
    "HierarchicalStitcher",
    "HierarchicalDOWStitcher",
    "SmoothAlphaStitcher",
]
