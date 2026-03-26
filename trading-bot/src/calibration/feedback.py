"""Calibration feedback: learn from settled contracts to improve future probabilities.

Uses Platt scaling (logistic regression) to map raw model probabilities
to empirically calibrated probabilities based on actual outcomes.
"""

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from ..data.logger import get_logger

log = get_logger("calibration_feedback")


@dataclass
class CalibrationModel:
    """Platt scaling parameters: calibrated_prob = sigmoid(a * raw_prob + b)"""
    a: float
    b: float
    n_samples: int
    brier_before: float
    brier_after: float
    city: str | None = None  # None = global model


def sigmoid(x: float) -> float:
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def fit_platt_scaling(
    predicted_probs: list[float],
    outcomes: list[int],
) -> tuple[float, float]:
    """Fit Platt scaling via gradient descent.

    Finds a, b such that sigmoid(a * p + b) best predicts outcomes.
    """
    a, b = 1.0, 0.0
    lr = 0.01

    for _ in range(1000):
        grad_a, grad_b = 0.0, 0.0
        for p, y in zip(predicted_probs, outcomes):
            q = sigmoid(a * p + b)
            error = q - y
            grad_a += error * p
            grad_b += error
        grad_a /= len(predicted_probs)
        grad_b /= len(predicted_probs)
        a -= lr * grad_a
        b -= lr * grad_b

    return a, b


def brier_score_binary(probs: list[float], outcomes: list[int]) -> float:
    """Brier score for binary outcomes. Lower = better."""
    return sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / len(probs)


def build_calibration_model(
    settled_contracts: list[dict],
    city: str | None = None,
    min_samples: int = 30,
) -> CalibrationModel | None:
    """Build a calibration model from settled contract data.

    Args:
        settled_contracts: list of dicts with 'model_prob' and 'result' fields
        city: if set, only use contracts from this city. None = global.
        min_samples: minimum settled contracts needed to train
    """
    data = settled_contracts
    if city:
        data = [d for d in data if d.get("city") == city]

    # Filter to contracts where we had a model prediction
    data = [d for d in data if d.get("model_prob") is not None]

    if len(data) < min_samples:
        log.info(
            "insufficient_calibration_data",
            city=city or "global",
            n_samples=len(data),
            min_required=min_samples,
        )
        return None

    probs = [d["model_prob"] for d in data]
    outcomes = [1 if d["result"] == "yes" else 0 for d in data]

    brier_before = brier_score_binary(probs, outcomes)

    a, b = fit_platt_scaling(probs, outcomes)

    calibrated = [sigmoid(a * p + b) for p in probs]
    brier_after = brier_score_binary(calibrated, outcomes)

    model = CalibrationModel(
        a=a, b=b,
        n_samples=len(data),
        brier_before=brier_before,
        brier_after=brier_after,
        city=city,
    )

    log.info(
        "calibration_model_trained",
        city=city or "global",
        n_samples=len(data),
        brier_before=f"{brier_before:.4f}",
        brier_after=f"{brier_after:.4f}",
        a=f"{a:.4f}",
        b=f"{b:.4f}",
    )
    return model


def apply_calibration(
    raw_probs: list[float],
    model: CalibrationModel,
) -> list[float]:
    """Apply Platt scaling to a list of raw model probabilities.

    Re-normalizes after calibration so probabilities sum to 1.
    """
    calibrated = [sigmoid(model.a * p + model.b) for p in raw_probs]
    total = sum(calibrated)
    if total > 0:
        calibrated = [p / total for p in calibrated]
    return calibrated


def save_calibration_model(
    model: CalibrationModel,
    model_dir: str = "./data/calibration_models/",
) -> None:
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    name = model.city.lower().replace(" ", "_") if model.city else "global"
    filepath = path / f"{name}_platt.json"
    with open(filepath, "w") as f:
        json.dump(asdict(model), f, indent=2)
    log.info("calibration_model_saved", path=str(filepath))


def load_calibration_model(
    city: str | None = None,
    model_dir: str = "./data/calibration_models/",
) -> CalibrationModel | None:
    """Load calibration model. Falls back to global if city-specific not found."""
    path = Path(model_dir)

    if city:
        city_file = path / f"{city.lower().replace(' ', '_')}_platt.json"
        if city_file.exists():
            with open(city_file) as f:
                return CalibrationModel(**json.load(f))

    global_file = path / "global_platt.json"
    if global_file.exists():
        with open(global_file) as f:
            return CalibrationModel(**json.load(f))

    return None


def generate_calibration_report(settled_contracts: list[dict]) -> dict:
    """Generate a calibration report with reliability stats.

    Groups predictions into probability bins and compares
    predicted probability vs actual outcome frequency.
    """
    data = [d for d in settled_contracts if d.get("model_prob") is not None]
    if not data:
        return {"error": "no data"}

    bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]

    report_bins = []
    for low, high in bins:
        bin_data = [d for d in data if low <= d["model_prob"] < high]
        if not bin_data:
            continue
        actual_rate = sum(1 for d in bin_data if d["result"] == "yes") / len(bin_data)
        avg_predicted = sum(d["model_prob"] for d in bin_data) / len(bin_data)
        report_bins.append({
            "range": f"{low:.0%}-{high:.0%}",
            "n": len(bin_data),
            "predicted": round(avg_predicted, 3),
            "actual": round(actual_rate, 3),
            "gap": round(avg_predicted - actual_rate, 3),
        })

    # Overall stats
    probs = [d["model_prob"] for d in data]
    outcomes = [1 if d["result"] == "yes" else 0 for d in data]

    # Trade P&L
    traded = [d for d in data if d.get("traded")]
    total_pnl = sum(d.get("pnl_cents", 0) or 0 for d in traded)

    return {
        "total_contracts": len(data),
        "total_traded": len(traded),
        "total_pnl_cents": total_pnl,
        "brier_score": round(brier_score_binary(probs, outcomes), 4),
        "bins": report_bins,
    }
