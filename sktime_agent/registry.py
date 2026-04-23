"""Curated registry of sktime estimators exposed to the LLM."""
from __future__ import annotations

import importlib
from typing import Any

FORECASTERS: dict[str, dict] = {
    "NaiveForecaster": {
        "module": "sktime.forecasting.naive",
        "description": "Baseline: last value, mean, or seasonal-last. Use as sanity-check baseline.",
        "params": {"strategy": "last|mean|seasonal_last", "sp": "seasonal period (int, default 1)"},
    },
    "ExponentialSmoothing": {
        "module": "sktime.forecasting.exp_smoothing",
        "description": "Holt-Winters exponential smoothing with optional trend and seasonality.",
        "params": {"trend": "add|mul|None", "seasonal": "add|mul|None", "sp": "seasonal period (int)"},
    },
    "AutoARIMA": {
        "module": "sktime.forecasting.arima",
        "description": "Auto-selects ARIMA order. Robust general-purpose forecaster.",
        "params": {"sp": "seasonal period (int)", "d": "differencing order (int|None)", "max_p": "max AR order (int)"},
    },
    "ARIMA": {
        "module": "sktime.forecasting.arima",
        "description": "ARIMA with manually specified order. Use when you know the exact order.",
        "params": {"order": "(p,d,q) tuple", "seasonal_order": "(P,D,Q,s) tuple"},
    },
    "ThetaForecaster": {
        "module": "sktime.forecasting.theta",
        "description": "Theta method. Simple, competitive on seasonal data, fast.",
        "params": {"sp": "seasonal period (int)", "deseasonalize": "bool (default True)"},
    },
    "AutoETS": {
        "module": "sktime.forecasting.ets",
        "description": "Auto ETS (Error/Trend/Seasonality) model selection.",
        "params": {"sp": "seasonal period (int)", "auto": "bool (default True)"},
    },
    "TBATS": {
        "module": "sktime.forecasting.tbats",
        "description": "TBATS for complex/multiple seasonality. Slower but handles unusual patterns.",
        "params": {"sp": "list of seasonal periods or single int"},
    },
}

TRANSFORMERS: dict[str, dict] = {
    "Deseasonalizer": {
        "module": "sktime.transformations.series.detrend",
        "description": "Remove seasonality via classical decomposition before forecasting.",
        "params": {"sp": "seasonal period (int)", "model": "additive|multiplicative"},
    },
    "Detrender": {
        "module": "sktime.transformations.series.detrend",
        "description": "Remove polynomial trend. Apply before a stationary forecaster.",
        "params": {"degree": "polynomial degree (int, default 1)"},
    },
    "BoxCoxTransformer": {
        "module": "sktime.transformations.series.boxcox",
        "description": "Box-Cox variance stabilisation. Use when variance grows with level.",
        "params": {"bounds": "(lower, upper) for lambda", "method": "mle|pearsonr"},
    },
    "LogTransformer": {
        "module": "sktime.transformations.series.boxcox",
        "description": "Log transform for exponential growth / heteroskedastic series.",
        "params": {},
    },
    "Imputer": {
        "module": "sktime.transformations.series.impute",
        "description": "Fill missing values before fitting.",
        "params": {"method": "mean|median|linear|nearest|bfill|ffill"},
    },
}


def get_estimator_class(name: str) -> Any:
    """Dynamically import and return an estimator class by name."""
    registry = {**FORECASTERS, **TRANSFORMERS}
    if name not in registry:
        raise ValueError(
            f"Unknown estimator '{name}'. "
            f"Available: {sorted(registry.keys())}"
        )
    module = importlib.import_module(registry[name]["module"])
    return getattr(module, name)
