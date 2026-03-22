from datetime import date

import numpy as np

from .bias_correction import load_bias_model, correct_ensemble, get_season
from .metar_client import fetch_todays_observations
from .metar_fusion import fuse_metar_with_ensemble, FusionResult
from .ensemble_calibration import compute_calibrated_bucket_probabilities
from .models import CalibratedForecast
from ..data.logger import get_logger

log = get_logger("calibration_pipeline")


async def run_calibration_pipeline(
    city: str,
    ensemble_temps_f: list[float],
    bucket_ranges: list[tuple[float | None, float | None]],
    metar_station: str | None = None,
    forecast_hour: int = 12,
    bias_model_dir: str = "./data/bias_models/",
    kde_bandwidth: float = 0.3,
    min_prob: float = 0.005,
    max_shift_f: float = 5.0,
) -> tuple[list[float], CalibratedForecast]:
    """Run full calibration pipeline: bias correction -> METAR fusion -> calibrated probs.

    Returns:
        (calibrated_probs, calibrated_forecast) tuple
    """
    raw = np.array(ensemble_temps_f)
    calibrated = np.copy(raw)
    bias_applied = False
    metar_applied = False
    metar_shift = None

    # Module 1: Station Bias Correction
    target_date = date.today()
    season = get_season(target_date)
    bias_model = load_bias_model(city, season, model_dir=bias_model_dir)
    if bias_model is not None:
        calibrated = correct_ensemble(calibrated, bias_model)
        bias_applied = True
        log.info(
            "bias_correction_applied",
            city=city,
            season=season,
            mae_before=bias_model.mae_before,
            mae_after=bias_model.mae_after,
        )

    # Module 2: Real-Time METAR Fusion
    if metar_station:
        try:
            observations = await fetch_todays_observations(metar_station)
            if observations:
                fusion: FusionResult = fuse_metar_with_ensemble(
                    calibrated,
                    observations,
                    forecast_hour=forecast_hour,
                    max_shift_f=max_shift_f,
                )
                calibrated = fusion.updated_ensemble
                metar_applied = True
                metar_shift = fusion.shift_applied
        except Exception as e:
            log.warning("metar_fusion_skipped", city=city, error=str(e))

    # Module 3: Calibrated Bucket Probabilities
    calibrated_probs = compute_calibrated_bucket_probabilities(
        calibrated,
        bucket_ranges,
        kde_bandwidth=kde_bandwidth,
        min_prob=min_prob,
    )

    forecast = CalibratedForecast(
        city=city,
        target_date=target_date,
        raw_ensemble=raw.tolist(),
        calibrated_ensemble=calibrated.tolist(),
        bias_correction_applied=bias_applied,
        metar_fusion_applied=metar_applied,
        metar_shift_f=metar_shift,
        ensemble_mean=float(np.mean(calibrated)),
        ensemble_std=float(np.std(calibrated)),
    )

    log.info(
        "calibration_complete",
        city=city,
        bias=bias_applied,
        metar=metar_applied,
        mean=f"{forecast.ensemble_mean:.1f}F",
        std=f"{forecast.ensemble_std:.1f}F",
    )

    return calibrated_probs, forecast
