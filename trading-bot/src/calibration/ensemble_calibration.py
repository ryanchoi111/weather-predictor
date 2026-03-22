import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import quad


def compute_calibrated_bucket_probabilities(
    ensemble_temps_f: np.ndarray,
    bucket_ranges: list[tuple[float | None, float | None]],
    kde_bandwidth: float = 0.3,
    min_prob: float = 0.005,
) -> list[float]:
    """Compute bucket probabilities using calibrated KDE bandwidth.

    Like the base probability module but with configurable bandwidth
    from ensemble calibration sweep results.
    """
    kde = gaussian_kde(ensemble_temps_f, bw_method=kde_bandwidth)

    raw_probs = []
    for low, high in bucket_ranges:
        low_bound = low if low is not None else float(np.min(ensemble_temps_f)) - 30
        high_bound = high if high is not None else float(np.max(ensemble_temps_f)) + 30
        prob, _ = quad(lambda x: kde(x).item(), low_bound, high_bound)
        raw_probs.append(max(prob, min_prob))

    total = sum(raw_probs)
    return [p / total for p in raw_probs]


def brier_score(
    predicted_probs: dict[str, float],
    actual_bucket: str,
) -> float:
    """Compute Brier Score. Lower is better. 0.0 = perfect."""
    score = 0.0
    for bucket, prob in predicted_probs.items():
        outcome = 1.0 if bucket == actual_bucket else 0.0
        score += (prob - outcome) ** 2
    return score / len(predicted_probs)
