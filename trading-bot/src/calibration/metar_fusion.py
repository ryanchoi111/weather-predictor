from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from .metar_client import METARObservation
from ..data.logger import get_logger

log = get_logger("metar_fusion")


@dataclass
class FusionResult:
    original_ensemble: np.ndarray
    updated_ensemble: np.ndarray
    observed_max_so_far: float
    observed_current: float | None
    shift_applied: float
    confidence: float
    n_observations: int


def fuse_metar_with_ensemble(
    ensemble_temps_f: np.ndarray,
    observations: list[METARObservation],
    forecast_hour: int,
    max_shift_f: float = 5.0,
    negative_shift_damping: float = 0.3,
) -> FusionResult:
    """Update ensemble based on real-time METAR observations.

    Rule 1 - Floor Constraint: daily high can't be lower than observed max.
    Rule 2 - Trend Adjustment: shift ensemble based on obs vs forecast deviation.
    """
    obs_temps = [o.temp_f for o in observations if o.is_valid]
    obs_max = max(obs_temps)
    obs_current = obs_temps[-1] if obs_temps else None
    ensemble_mean = float(np.mean(ensemble_temps_f))

    updated = np.copy(ensemble_temps_f)

    # Rule 1: Floor constraint
    updated = np.maximum(updated, obs_max)

    # Rule 2: Trend adjustment
    hour_confidence = min(forecast_hour / 18.0, 1.0)
    shift = (obs_max - ensemble_mean) * hour_confidence * 0.5

    if shift < 0:
        shift *= negative_shift_damping

    shift = np.clip(shift, -max_shift_f, max_shift_f)
    updated = updated + shift

    log.info(
        "metar_fusion_applied",
        obs_max=f"{obs_max:.1f}F",
        ensemble_mean=f"{ensemble_mean:.1f}F",
        shift=f"{shift:.1f}F",
        confidence=f"{hour_confidence:.2f}",
        n_obs=len(observations),
    )

    return FusionResult(
        original_ensemble=ensemble_temps_f,
        updated_ensemble=updated,
        observed_max_so_far=obs_max,
        observed_current=obs_current,
        shift_applied=float(shift),
        confidence=hour_confidence,
        n_observations=len(observations),
    )


def get_nws_high_temp_window(
    target_date: date, timezone: str
) -> tuple[datetime, datetime]:
    """Returns (start, end) for the NWS high temp reporting window, DST-aware."""
    tz = ZoneInfo(timezone)
    noon = datetime.combine(target_date, time(12, 0), tzinfo=tz)
    is_dst = bool(noon.dst())

    if is_dst:
        start = datetime.combine(target_date, time(1, 0), tzinfo=tz)
        end = datetime.combine(target_date + timedelta(days=1), time(0, 59), tzinfo=tz)
    else:
        start = datetime.combine(target_date, time(0, 0), tzinfo=tz)
        end = datetime.combine(target_date, time(23, 59), tzinfo=tz)

    return start, end
