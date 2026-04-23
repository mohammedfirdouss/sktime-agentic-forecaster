import os
from dotenv import load_dotenv
load_dotenv()

from sktime.datasets import load_airline
from sktime_agent import AgenticForecaster

y = load_airline()
y_train = y[:-12]

result = AgenticForecaster(llm_backend="gemini", verbose=True).forecast(
    prompt="Forecast 12 months of airline passengers. Strong trend and annual seasonality.",
    data=y_train,
    horizon=12,
)

print("\n=== Result ===")
print("Pipeline:", result.selected_estimators)
print("Explanation:", result.explanation)
print("Predictions:\n", result.predictions)
