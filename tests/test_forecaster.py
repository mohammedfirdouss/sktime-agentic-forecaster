"""Tests for AgenticForecaster — LLM calls are mocked."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from sktime_agent import AgenticForecaster, ForecastResult


@pytest.fixture
def airline():
    from sktime.datasets import load_airline
    return load_airline()


def _spec(transformers, forecaster_class, forecaster_params=None, explanation="test"):
    return json.dumps({
        "transformers": transformers,
        "forecaster": {"class": forecaster_class, "params": forecaster_params or {}},
        "explanation": explanation,
    })


def _mock_forecaster(backend_name: str, raw_response: str) -> AgenticForecaster:
    af = AgenticForecaster(llm_backend=backend_name)
    af._backend = MagicMock()
    af._backend.complete.return_value = raw_response
    return af


class TestBasicForecast:
    def test_naive_no_transforms(self, airline):
        af = _mock_forecaster("openai", _spec([], "NaiveForecaster", {"strategy": "last"}))
        result = af.forecast("Forecast 12 months ahead", airline, horizon=12)

        assert isinstance(result, ForecastResult)
        assert len(result.predictions) == 12
        assert result.selected_estimators == ["NaiveForecaster"]

    def test_theta_with_sp(self, airline):
        af = _mock_forecaster(
            "anthropic",
            _spec([], "ThetaForecaster", {"sp": 12}, "Theta for monthly data"),
        )
        result = af.forecast("Seasonal monthly data", airline, horizon=6)
        assert len(result.predictions) == 6
        assert result.explanation == "Theta for monthly data"

    def test_pipeline_with_transforms(self, airline):
        raw = _spec(
            [
                {"class": "Deseasonalizer", "params": {"sp": 12}},
                {"class": "Detrender", "params": {}},
            ],
            "AutoARIMA",
            {"sp": 12},
            "Deseasonalise, detrend, then ARIMA",
        )
        af = _mock_forecaster("openai", raw)
        result = af.forecast(
            "Forecast with deseasonalisation and detrending", airline, horizon=12
        )
        assert len(result.predictions) == 12
        assert "Deseasonalizer" in result.selected_estimators
        assert "Detrender" in result.selected_estimators
        assert "AutoARIMA" in result.selected_estimators

    def test_dataframe_single_column(self, airline):
        df = airline.to_frame()
        af = _mock_forecaster("openai", _spec([], "NaiveForecaster"))
        result = af.forecast("test", df, horizon=3)
        assert len(result.predictions) == 3

    def test_dataframe_multi_column_raises(self, airline):
        import numpy as np
        df = pd.DataFrame(
            {"a": airline.values, "b": airline.values},
            index=airline.index,
        )
        af = _mock_forecaster("openai", _spec([], "NaiveForecaster"))
        with pytest.raises(ValueError, match="Multi-variate"):
            af.forecast("test", df, horizon=3)


class TestErrorHandling:
    def test_invalid_json_raises(self, airline):
        af = _mock_forecaster("openai", "not valid json {{{{")
        with pytest.raises(ValueError, match="invalid JSON"):
            af.forecast("test", airline, horizon=3)

    def test_unknown_estimator_raises(self, airline):
        raw = _spec([], "NonExistentForecaster9000")
        af = _mock_forecaster("openai", raw)
        with pytest.raises(ValueError, match="Unknown estimator"):
            af.forecast("test", airline, horizon=3)

    def test_markdown_fenced_json(self, airline):
        inner = {"transformers": [], "forecaster": {"class": "NaiveForecaster", "params": {}}, "explanation": "ok"}
        wrapped = f"```json\n{json.dumps(inner)}\n```"
        af = _mock_forecaster("openai", wrapped)
        result = af.forecast("test", airline, horizon=3)
        assert len(result.predictions) == 3


class TestRepr:
    def test_result_repr(self, airline):
        af = _mock_forecaster("openai", _spec([], "NaiveForecaster"))
        result = af.forecast("test", airline, horizon=3)
        assert "ForecastResult" in repr(result)
        assert "NaiveForecaster" in repr(result)
