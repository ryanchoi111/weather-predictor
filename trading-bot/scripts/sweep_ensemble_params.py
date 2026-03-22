"""Grid search over ensemble calibration parameters to minimize Brier Score.

Usage:
    python -m scripts.sweep_ensemble_params --all-cities
    python -m scripts.sweep_ensemble_params --station KNYC

Requires NWS history data from fetch_nws_history.py.
Uses historical data to simulate ensemble forecasts and evaluate
Brier Score across KDE bandwidth values.
"""

import argparse
import csv
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from src.calibration.ensemble_calibration import (
    compute_calibrated_bucket_probabilities,
    brier_score,
)


# Typical Kalshi bucket ranges (2F wide from 30F to 110F)
DEFAULT_BUCKETS = (
    [(None, 30.0)]
    + [(float(t), float(t + 2)) for t in range(30, 110, 2)]
    + [(110.0, None)]
)


def load_nws_history(station: str, data_dir: str = "./data/nws_history/") -> list[dict]:
    path = Path(data_dir) / f"{station.lower()}_history.csv"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                records.append({
                    "date": date.fromisoformat(row["date"]),
                    "max_temp_f": float(row["max_temp_f"]),
                })
            except (ValueError, KeyError):
                continue
    return records


def find_actual_bucket(temp_f: float, buckets: list[tuple[float | None, float | None]]) -> int:
    """Find which bucket index a temperature falls into."""
    for i, (low, high) in enumerate(buckets):
        low_ok = low is None or temp_f >= low
        high_ok = high is None or temp_f < high
        if low_ok and high_ok:
            return i
    return len(buckets) - 1


def simulate_ensemble(actual_temp: float, n_members: int = 10, std: float = 3.0) -> np.ndarray:
    """Simulate an ensemble forecast centered near actual temp with noise."""
    return np.random.normal(actual_temp, std, n_members)


def sweep_bandwidth(
    records: list[dict],
    bandwidths: list[float],
    buckets: list[tuple[float | None, float | None]],
    n_members: int = 10,
    ensemble_std: float = 3.0,
    max_samples: int = 100,
) -> dict[float, float]:
    """Evaluate each bandwidth on historical data. Returns {bandwidth: mean_brier}.

    Subsamples records to max_samples for speed (KDE+quad is expensive).
    """
    if len(records) > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(records), max_samples, replace=False)
        sampled = [records[i] for i in indices]
    else:
        sampled = records

    results = {}
    for bw in bandwidths:
        scores = []
        for record in sampled:
            actual = record["max_temp_f"]
            ensemble = simulate_ensemble(actual, n_members, ensemble_std)
            probs = compute_calibrated_bucket_probabilities(ensemble, buckets, kde_bandwidth=bw)

            actual_idx = find_actual_bucket(actual, buckets)
            bucket_labels = {str(i): p for i, p in enumerate(probs)}
            bs = brier_score(bucket_labels, str(actual_idx))
            scores.append(bs)

        results[bw] = float(np.mean(scores))

    return results


def load_cities(config_path: str = "config/cities.yaml") -> list[dict]:
    with open(config_path) as f:
        return yaml.safe_load(f).get("cities", [])


def main():
    parser = argparse.ArgumentParser(description="Sweep ensemble KDE bandwidth")
    parser.add_argument("--station", type=str)
    parser.add_argument("--all-cities", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./data/sweep_results/")
    args = parser.parse_args()

    bandwidths = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0]

    def run_sweep(city_name: str, station: str):
        records = load_nws_history(station)
        if not records:
            print(f"  No history for {station}")
            return
        print(f"  Sweeping {len(bandwidths)} bandwidths over {len(records)} days...")
        results = sweep_bandwidth(records, bandwidths, DEFAULT_BUCKETS)

        best_bw = min(results, key=results.get)
        print(f"  Results:")
        for bw, score in sorted(results.items()):
            marker = " <-- BEST" if bw == best_bw else ""
            print(f"    bw={bw:.2f}: Brier={score:.6f}{marker}")

        # Save results
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{station.lower()}_sweep.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bandwidth", "mean_brier_score"])
            for bw, score in sorted(results.items()):
                writer.writerow([bw, score])
        print(f"  Saved to {out_path}")

    if args.all_cities:
        cities = load_cities()
        for city in cities:
            print(f"\n{city['name']} ({city['nws_station']}):")
            run_sweep(city["name"], city["nws_station"])
    elif args.station:
        run_sweep(args.station, args.station)
    else:
        parser.error("Specify --station or --all-cities")


if __name__ == "__main__":
    main()
