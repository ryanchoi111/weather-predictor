"""Fetch historical NWS daily climate data from Iowa Environmental Mesonet.

Usage:
    python -m scripts.fetch_nws_history --station KNYC --years 2
    python -m scripts.fetch_nws_history --all-cities --years 2
"""

import argparse
import csv
import io
from datetime import date, timedelta
from pathlib import Path

import httpx
import yaml


IEM_CLI_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"

# Map ICAO prefix to IEM state network
STATION_NETWORKS = {
    "NYC": "NY_ASOS", "JFK": "NY_ASOS", "LGA": "NY_ASOS",
    "ORD": "IL_ASOS", "MDW": "IL_ASOS",
    "LAX": "CA_ASOS", "SFO": "CA_ASOS",
    "MIA": "FL_ASOS", "FLL": "FL_ASOS",
    "AUS": "TX_ASOS", "DFW": "TX_ASOS", "IAH": "TX_ASOS",
}


def _guess_networks(station_id: str) -> list[str]:
    """Return candidate IEM networks for a station ID."""
    candidates = []
    if station_id in STATION_NETWORKS:
        candidates.append(STATION_NETWORKS[station_id])
    # Try state prefix guess (first 2 chars)
    candidates.append(f"{station_id[:2]}_ASOS")
    candidates.append("AWOS")
    return list(dict.fromkeys(candidates))  # dedupe preserving order


def fetch_daily_temps(station: str, start: date, end: date) -> list[dict]:
    """Fetch daily max/min temps from IEM for a station."""
    station_id = station.lstrip("K")
    params = {
        "stations": station_id,
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
        "format": "comma",
    }

    for network in _guess_networks(station_id):
        params["network"] = network
        try:
            resp = httpx.get(IEM_CLI_URL, params=params, timeout=30)
            resp.raise_for_status()
            records = _parse_csv(resp.text, station)
            if records:
                print(f"  Found {len(records)} records via network={network}")
                return records
        except Exception:
            continue

    print(f"  WARNING: No data found for {station}")
    return []


def _parse_csv(text: str, station: str) -> list[dict]:
    """Parse IEM CSV response into records."""
    records = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            max_tmpf = row.get("max_temp_f", "M")
            if max_tmpf in ("M", "None", "") or not max_tmpf:
                continue
            records.append({
                "station": station,
                "date": row["day"],
                "max_temp_f": float(max_tmpf),
            })
        except (ValueError, KeyError):
            continue
    return records


def save_records(records: list[dict], output_dir: Path, station: str) -> None:
    """Save records as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{station.lower()}_history.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["station", "date", "max_temp_f"])
        writer.writeheader()
        writer.writerows(records)
    print(f"  Saved {len(records)} records to {path}")


def load_cities(config_path: str = "config/cities.yaml") -> list[dict]:
    with open(config_path) as f:
        return yaml.safe_load(f).get("cities", [])


def main():
    parser = argparse.ArgumentParser(description="Fetch NWS historical temps from IEM")
    parser.add_argument("--station", type=str, help="ICAO station code (e.g. KNYC)")
    parser.add_argument("--all-cities", action="store_true", help="Fetch for all cities in config")
    parser.add_argument("--years", type=int, default=2, help="Years of history to fetch")
    parser.add_argument("--output-dir", type=str, default="./data/nws_history/")
    args = parser.parse_args()

    end = date.today()
    start = date(end.year - args.years, end.month, end.day)
    output_dir = Path(args.output_dir)

    if args.all_cities:
        cities = load_cities()
        for city in cities:
            station = city["nws_station"]
            print(f"Fetching {city['name']} ({station})...")
            records = fetch_daily_temps(station, start, end)
            if records:
                save_records(records, output_dir, station)
    elif args.station:
        print(f"Fetching {args.station}...")
        records = fetch_daily_temps(args.station, start, end)
        if records:
            save_records(records, output_dir, args.station)
    else:
        parser.error("Specify --station or --all-cities")


if __name__ == "__main__":
    main()
