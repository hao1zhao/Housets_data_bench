from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

from housets_bench.bundles.datatypes import ProcBundle


class BaseForecaster(ABC):
    name: str = "base"

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        """Optional fit step. Default: no-op."""
        return

    @abstractmethod
    def predict_batch(
        self,
        batch: Dict[str, Any],
        *,
        bundle: ProcBundle,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Predict the next horizon for a single batch in processed space."""
        raise NotImplementedError
