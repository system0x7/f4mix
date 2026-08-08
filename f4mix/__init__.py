"""Sample-wise covariance-aware f4 profile fitting.

The public API is a run-script-friendly trio: :func:`open_genotypes` for
metadata-only sample selection, :class:`F4Model` for fitting, and
:class:`F4FitResult` for inspecting and saving results.  The numerical core
 lives in ``f4mix.core``.
"""

from .api import (
    F4FitConfig,
    F4FitResult,
    F4Model,
)
from .dataset import F4Dataset, open_genotypes

__all__ = [
    "F4Dataset",
    "F4FitConfig",
    "F4FitResult",
    "F4Model",
    "open_genotypes",
]
