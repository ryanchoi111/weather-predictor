"""RunPod serverless handler wrapping the forecast engine."""

import os
import numpy as np
import runpod

from .schemas import ForecastRequest, ForecastResponse

MOCK_MODE = os.environ.get("MOCK_MODE", "true").lower() == "true"


def _mock_forecast(req: ForecastRequest) -> dict:
    base_temp = 75 - (req.latitude - 25) * 0.8
    temps = np.random.normal(loc=base_temp, scale=4.0, size=req.nensemble).tolist()
    return ForecastResponse(
        temperatures_f=temps,
        mean_f=float(np.mean(temps)),
        std_f=float(np.std(temps)),
        model_name=req.model_name,
        lead_hours=req.lead_hours,
    ).model_dump()


async def _real_forecast(req: ForecastRequest) -> dict:
    from .forecast.engine import run_ensemble
    from .forecast.extract import extract_point_temperature

    output = await run_ensemble(
        model_name=req.model_name,
        nensemble=req.nensemble,
    )
    temps_f = extract_point_temperature(output, req.latitude, req.longitude)
    return ForecastResponse(
        temperatures_f=temps_f,
        mean_f=float(np.mean(temps_f)),
        std_f=float(np.std(temps_f)),
        model_name=req.model_name,
        lead_hours=req.lead_hours,
    ).model_dump()


async def handler(job: dict) -> dict:
    """RunPod serverless handler. Input matches ForecastRequest schema."""
    try:
        req = ForecastRequest(**job["input"])
    except Exception as e:
        return {"error": f"Invalid input: {e}"}

    try:
        if MOCK_MODE:
            return _mock_forecast(req)
        return await _real_forecast(req)
    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
