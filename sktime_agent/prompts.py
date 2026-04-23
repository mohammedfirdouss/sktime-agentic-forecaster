"""Prompt templates for estimator selection."""
from __future__ import annotations

import json

from .registry import FORECASTERS, TRANSFORMERS

SYSTEM_PROMPT = """\
You are an expert time series forecasting engineer with deep knowledge of sktime.
Given a forecasting task description and dataset properties, you select the best
pipeline of sktime estimators and configure their hyperparameters appropriately.

You always return valid JSON and nothing else — no markdown, no prose outside the object.\
"""


def build_selection_prompt(
    user_prompt: str,
    n_obs: int,
    frequency: str | None,
    has_missing: bool,
) -> str:
    forecasters_desc = json.dumps(
        {k: {"description": v["description"], "params": v["params"]} for k, v in FORECASTERS.items()},
        indent=2,
    )
    transformers_desc = json.dumps(
        {k: {"description": v["description"], "params": v["params"]} for k, v in TRANSFORMERS.items()},
        indent=2,
    )

    return f"""\
Forecasting task: {user_prompt}

Dataset properties:
  observations : {n_obs}
  frequency    : {frequency or "unknown"}
  missing vals : {has_missing}

Available forecasters:
{forecasters_desc}

Available transformers (prepended to the pipeline before the forecaster):
{transformers_desc}

Return ONLY this JSON — no text outside it:
{{
  "transformers": [
    {{"class": "TransformerName", "params": {{}}}}
  ],
  "forecaster": {{"class": "ForecasterName", "params": {{}}}},
  "explanation": "Why you chose this pipeline"
}}

Guidelines:
- transformers may be an empty list []
- forecaster must be exactly one class from the Available forecasters list
- use only class names exactly as listed above
- set sp (seasonal period) from frequency: monthly=12, quarterly=4, weekly=52, daily=7, hourly=24
- if the prompt mentions deseasonalising or detrending, include those transformers
- for short series (< 50 obs) prefer simpler models (NaiveForecaster, ThetaForecaster)
- if data has missing values, prepend an Imputer transformer
"""
