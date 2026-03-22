import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import quad


def compute_bucket_probabilities(
    ensemble_temps: list[float],
    buckets: list[tuple[float | None, float | None]],
    min_prob: float = 0.005,
) -> list[float]:
    """Compute probability for each temperature bucket using KDE.

    Args:
        ensemble_temps: ensemble forecast temperatures (Fahrenheit)
        buckets: list of (low, high) tuples. None = unbounded.
        min_prob: minimum probability floor per bucket

    Returns:
        Normalized probabilities summing to 1.0
    """
    kde = gaussian_kde(ensemble_temps, bw_method="silverman")

    raw_probs = []
    for low, high in buckets:
        low_bound = low if low is not None else min(ensemble_temps) - 30
        high_bound = high if high is not None else max(ensemble_temps) + 30
        prob, _ = quad(lambda x: kde(x).item(), low_bound, high_bound)
        raw_probs.append(max(prob, min_prob))

    total = sum(raw_probs)
    return [p / total for p in raw_probs]
