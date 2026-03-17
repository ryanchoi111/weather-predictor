import os
import numpy as np
from fastapi import FastAPI, HTTPException, Header

from .schemas import ForecastRequest, ForecastResponse

app = FastAPI(title="Weather Forecast Service")

API_KEY = os.environ.get("API_KEY", "")
MOCK_MODE = os.environ.get("MOCK_MODE", "true").lower() == "true"


def _verify_api_key(authorization: str | None = Header(None)) -> None:
    if not API_KEY:
        return
    if not authorization or authorization.replace("Bearer ", "") != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest, authorization: str | None = Header(None)):
    _verify_api_key(authorization)

    if MOCK_MODE:
        return _mock_forecast(request)

    try:
        from .forecast.engine import run_ensemble
        from .forecast.extract import extract_point_temperature

        output = await run_ensemble(
            model_name=request.model_name,
            nensemble=request.nensemble,
        )
        temps_f = extract_point_temperature(
            output, request.latitude, request.longitude,
        )
        return ForecastResponse(
            temperatures_f=temps_f,
            mean_f=float(np.mean(temps_f)),
            std_f=float(np.std(temps_f)),
            model_name=request.model_name,
            lead_hours=request.lead_hours,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _mock_forecast(request: ForecastRequest) -> ForecastResponse:
    base_temp = 75 - (request.latitude - 25) * 0.8
    temps = np.random.normal(loc=base_temp, scale=4.0, size=request.nensemble).tolist()
    return ForecastResponse(
        temperatures_f=temps,
        mean_f=float(np.mean(temps)),
        std_f=float(np.std(temps)),
        model_name=request.model_name,
        lead_hours=request.lead_hours,
    )


@app.get("/health")
async def health():
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass
    return {"status": "ok", "gpu_available": gpu_available, "mock_mode": MOCK_MODE}


@app.get("/models")
async def models():
    from .forecast.models import AVAILABLE_MODELS
    return {"models": AVAILABLE_MODELS}
