"""Train seasonal bias models using historical NWS data + model hindcasts.

Usage:
    python -m scripts.train_bias_models --station KNYC
    python -m scripts.train_bias_models --all-cities

Requires NWS history data from fetch_nws_history.py first.
Without model hindcast data, trains an identity mapping as baseline.
"""

import argparse
import csv
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from src.calibration.bias_correction import (
    BiasModel,
    get_season,
    train_quantile_mapping,
    save_bias_model,
)


SEASONS = ["DJF", "MAM", "JJA", "SON"]


def load_nws_history(station: str, data_dir: str = "./data/nws_history/") -> list[dict]:
    """Load NWS historical records from CSV."""
    path = Path(data_dir) / f"{station.lower()}_history.csv"
    if not path.exists():
        print(f"  No history file: {path}")
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


def load_model_hindcasts(station: str, data_dir: str = "./data/hindcasts/") -> list[dict]:
    """Load model hindcast data if available.

    Expected CSV format: date,ensemble_mean
    If not available, returns empty list and we use NWS data as both
    model and truth (identity baseline).
    """
    path = Path(data_dir) / f"{station.lower()}_hindcasts.csv"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                records.append({
                    "date": date.fromisoformat(row["date"]),
                    "ensemble_mean": float(row["ensemble_mean"]),
                })
            except (ValueError, KeyError):
                continue
    return records


def train_for_station(city_name: str, station: str, model_dir: str = "./data/bias_models/") -> None:
    """Train seasonal bias models for one station."""
    nws_data = load_nws_history(station)
    if not nws_data:
        print(f"  Skipping {station}: no NWS history")
        return

    hindcasts = load_model_hindcasts(station)
    hindcast_map = {r["date"]: r["ensemble_mean"] for r in hindcasts}

    for season in SEASONS:
        # Filter NWS records for this season
        season_nws = [r for r in nws_data if get_season(r["date"]) == season]
        if len(season_nws) < 30:
            print(f"  {season}: only {len(season_nws)} samples, skipping (need 30+)")
            continue

        nws_temps = np.array([r["max_temp_f"] for r in season_nws])

        if hindcast_map:
            # Use actual model hindcasts
            paired = [(r["max_temp_f"], hindcast_map[r["date"]])
                      for r in season_nws if r["date"] in hindcast_map]
            if len(paired) < 30:
                print(f"  {season}: only {len(paired)} paired samples, using identity baseline")
                model_temps = nws_temps + np.random.normal(0, 0.5, len(nws_temps))
            else:
                nws_temps = np.array([p[0] for p in paired])
                model_temps = np.array([p[1] for p in paired])
        else:
            # No hindcasts: create near-identity baseline with small noise
            model_temps = nws_temps + np.random.normal(0, 0.5, len(nws_temps))

        model_q, nws_q = train_quantile_mapping(model_temps, nws_temps)

        # Compute MAE before/after
        mae_before = float(np.mean(np.abs(model_temps - nws_temps)))
        # After correction, quantile-mapped values should be closer
        from src.calibration.bias_correction import apply_quantile_mapping
        corrected = np.array([apply_quantile_mapping(t, model_q, nws_q) for t in model_temps])
        mae_after = float(np.mean(np.abs(corrected - nws_temps)))

        bias = BiasModel(
            city=city_name,
            season=season,
            method="quantile_mapping",
            model_quantiles=model_q,
            nws_quantiles=nws_q,
            linear_slope=1.0,
            linear_intercept=0.0,
            n_samples=len(nws_temps),
            mae_before=mae_before,
            mae_after=mae_after,
        )
        save_bias_model(bias, model_dir=model_dir)
        print(f"  {season}: {len(nws_temps)} samples, MAE {mae_before:.2f}F -> {mae_after:.2f}F")


def load_cities(config_path: str = "config/cities.yaml") -> list[dict]:
    with open(config_path) as f:
        return yaml.safe_load(f).get("cities", [])


def main():
    parser = argparse.ArgumentParser(description="Train seasonal bias correction models")
    parser.add_argument("--station", type=str, help="ICAO station code")
    parser.add_argument("--all-cities", action="store_true", help="Train for all configured cities")
    parser.add_argument("--model-dir", type=str, default="./data/bias_models/")
    args = parser.parse_args()

    if args.all_cities:
        cities = load_cities()
        for city in cities:
            print(f"Training {city['name']} ({city['nws_station']})...")
            train_for_station(city["name"], city["nws_station"], args.model_dir)
    elif args.station:
        print(f"Training {args.station}...")
        train_for_station(args.station, args.station, args.model_dir)
    else:
        parser.error("Specify --station or --all-cities")


if __name__ == "__main__":
    main()
