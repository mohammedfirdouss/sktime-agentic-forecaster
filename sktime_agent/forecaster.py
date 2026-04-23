"""AgenticForecaster — main entry point."""
from __future__ import annotations

import json
import warnings
from typing import Any

import pandas as pd

from .llm_backend import LLMBackend, create_backend
from .pipeline_builder import build_pipeline, get_estimator_names, parse_llm_response
from .prompts import SYSTEM_PROMPT, build_selection_prompt
from .result import ForecastResult


def _detect_frequency(data: pd.Series) -> str | None:
    if not isinstance(data.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        return None
    try:
        return pd.infer_freq(data.index)
    except Exception:
        return None


def _make_fh(horizon: int) -> Any:
    from sktime.forecasting.base import ForecastingHorizon
    return ForecastingHorizon(list(range(1, horizon + 1)))


class AgenticForecaster:
    """LLM-powered forecasting agent built on sktime.

    Parameters
    ----------
    llm_backend : str or LLM object
        ``"openai"``, ``"anthropic"``, ``"gemini"``, or any LangChain LLM.
    verbose : bool
        Print selected pipeline and reasoning when True.
    """

    def __init__(self, llm_backend: str | Any = "openai", verbose: bool = False):
        self._backend: LLMBackend = create_backend(llm_backend)
        self.verbose = verbose
        self._last_spec: dict | None = None

    def forecast(
        self,
        prompt: str,
        data: pd.Series | pd.DataFrame,
        horizon: int,
    ) -> ForecastResult:
        """Forecast ``horizon`` steps ahead using an LLM-selected pipeline.

        Parameters
        ----------
        prompt : str
            Natural language description of the forecasting task.
        data : pd.Series or single-column pd.DataFrame
            Training data with a DatetimeIndex or PeriodIndex.
        horizon : int
            Number of steps ahead to forecast.

        Returns
        -------
        ForecastResult
            ``.predictions`` — pd.Series of forecasted values
            ``.pipeline``    — fitted sktime estimator / pipeline
            ``.explanation`` — LLM's reasoning for the choice
            ``.selected_estimators`` — list of class names used
        """
        if isinstance(data, pd.DataFrame):
            if data.shape[1] != 1:
                raise ValueError(
                    "Multi-variate forecasting is not yet supported. "
                    "Pass a Series or a single-column DataFrame."
                )
            data = data.iloc[:, 0]

        n_obs = len(data)
        frequency = _detect_frequency(data)
        has_missing = bool(data.isna().any())

        user_msg = build_selection_prompt(
            user_prompt=prompt,
            n_obs=n_obs,
            frequency=frequency,
            has_missing=has_missing,
        )

        if self.verbose:
            print(f"[AgenticForecaster] Querying {type(self._backend).__name__}…")

        raw = self._backend.complete(system=SYSTEM_PROMPT, user=user_msg)

        try:
            spec = parse_llm_response(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM returned invalid JSON.\nRaw response:\n{raw}"
            ) from exc

        self._last_spec = spec
        explanation = spec.get("explanation", "")
        names = get_estimator_names(spec)

        if self.verbose:
            print(f"[AgenticForecaster] Pipeline: {' → '.join(names)}")
            print(f"[AgenticForecaster] Reasoning: {explanation}")

        pipeline = build_pipeline(spec)
        fh = _make_fh(horizon)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(data)
            predictions = pipeline.predict(fh)

        return ForecastResult(
            predictions=predictions,
            pipeline=pipeline,
            explanation=explanation,
            selected_estimators=names,
        )
