import json
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from ..data.logger import get_logger

log = get_logger("bias_correction")


@dataclass
class BiasModel:
    city: str
    season: str
    method: str
    model_quantiles: list
    nws_quantiles: list
    linear_slope: float
    linear_intercept: float
    n_samples: int
    mae_before: float
    mae_after: float


def get_season(target_date: date) -> str:
    month = target_date.month
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def train_quantile_mapping(
    model_temps: np.ndarray, nws_temps: np.ndarray
) -> tuple[list, list]:
    """Train quantile mapping from model forecasts to NWS observations."""
    percentiles = np.linspace(0, 100, 101)
    model_q = np.percentile(model_temps, percentiles)
    nws_q = np.percentile(nws_temps, percentiles)
    return model_q.tolist(), nws_q.tolist()


def apply_quantile_mapping(
    raw_temp_f: float, model_q: list, nws_q: list
) -> float:
    """Apply quantile mapping to correct a single temperature value."""
    mapper = interp1d(
        model_q, nws_q,
        bounds_error=False,
        fill_value=(nws_q[0], nws_q[-1]),
    )
    return float(mapper(raw_temp_f))


def correct_ensemble(
    ensemble_temps_f: np.ndarray, bias_model: BiasModel
) -> np.ndarray:
    """Apply bias correction to every ensemble member."""
    if bias_model.method == "quantile_mapping":
        return np.array([
            apply_quantile_mapping(t, bias_model.model_quantiles, bias_model.nws_quantiles)
            for t in ensemble_temps_f
        ])
    return ensemble_temps_f * bias_model.linear_slope + bias_model.linear_intercept


def load_bias_model(
    city: str, season: str, model_dir: str = "./data/bias_models/"
) -> BiasModel | None:
    """Load a bias model from JSON. Returns None if not found."""
    path = Path(model_dir) / f"{city.lower()}_{season}.json"
    if not path.exists():
        log.info("bias_model_not_found", city=city, season=season)
        return None
    with open(path) as f:
        data = json.load(f)
    return BiasModel(**data)


def save_bias_model(model: BiasModel, model_dir: str = "./data/bias_models/") -> None:
    """Save a bias model to JSON."""
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / f"{model.city.lower()}_{model.season}.json"
    with open(filepath, "w") as f:
        json.dump(asdict(model), f, indent=2)
    log.info("bias_model_saved", path=str(filepath))
