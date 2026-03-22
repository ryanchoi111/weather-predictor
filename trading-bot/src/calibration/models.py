from datetime import date
from pydantic import BaseModel


class CalibratedForecast(BaseModel):
    city: str
    target_date: date
    raw_ensemble: list[float]
    calibrated_ensemble: list[float]
    bias_correction_applied: bool = False
    metar_fusion_applied: bool = False
    metar_shift_f: float | None = None
    ensemble_mean: float
    ensemble_std: float
