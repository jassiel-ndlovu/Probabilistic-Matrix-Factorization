"""Model implementations.

``TorchPMF`` is imported lazily so the package remains usable without PyTorch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import History, Recommender
from .bpmf import BPMF
from .pmf import PMF

if TYPE_CHECKING:  # pragma: no cover
    from .torch_pmf import TorchPMF

__all__ = ["BPMF", "PMF", "History", "Recommender", "TorchPMF"]


def __getattr__(name: str) -> Any:
    if name == "TorchPMF":
        from .torch_pmf import TorchPMF

        return TorchPMF
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
