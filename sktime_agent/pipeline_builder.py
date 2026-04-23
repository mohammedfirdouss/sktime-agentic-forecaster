"""Build a sktime pipeline from the LLM's JSON spec."""
from __future__ import annotations

import json
from typing import Any

from .registry import get_estimator_class


def parse_llm_response(text: str) -> dict:
    """Parse LLM output, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # drop opening fence (```json or ```) and closing fence
        inner = lines[1:] if lines[-1].strip() == "```" else lines[1:-1]
        text = "\n".join(inner)
    return json.loads(text)


def build_pipeline(spec: dict) -> Any:
    """Instantiate a sktime pipeline from a parsed LLM spec."""
    transformers = spec.get("transformers", [])
    forecaster_spec = spec["forecaster"]

    forecaster_cls = get_estimator_class(forecaster_spec["class"])
    forecaster = forecaster_cls(**forecaster_spec.get("params", {}))

    if not transformers:
        return forecaster

    from sktime.forecasting.compose import TransformedTargetForecaster

    steps = []
    for t in transformers:
        cls = get_estimator_class(t["class"])
        steps.append((t["class"].lower(), cls(**t.get("params", {}))))
    steps.append(("forecaster", forecaster))

    return TransformedTargetForecaster(steps)


def get_estimator_names(spec: dict) -> list[str]:
    names = [t["class"] for t in spec.get("transformers", [])]
    names.append(spec["forecaster"]["class"])
    return names
