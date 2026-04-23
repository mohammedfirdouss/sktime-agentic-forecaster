from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ForecastResult:
    predictions: pd.Series
    pipeline: Any
    explanation: str
    selected_estimators: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"ForecastResult(\n"
            f"  predictions={self.predictions.shape},\n"
            f"  pipeline={type(self.pipeline).__name__},\n"
            f"  estimators={self.selected_estimators}\n"
            f")"
        )
