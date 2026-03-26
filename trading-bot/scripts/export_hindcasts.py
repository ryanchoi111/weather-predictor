"""Backfill actual highs from NWS and export forecast-vs-actual pairs for bias retraining.

Usage:
    python -m scripts.export_hindcasts --backfill   # fill in missing actuals from NWS API
    python -m scripts.export_hindcasts --export      # export paired CSVs to data/hindcasts/
    python -m scripts.export_hindcasts --backfill --export  # both

After exporting, retrain bias models:
    python -m scripts.train_bias_models --all-cities
"""

import argparse
import asyncio
import csv
import io
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import aiosqlite

DB_PATH = Path(__file__).parent.parent / "data" / "trading.db"
NWS_API = "https://api.weather.gov"
USER_AGENT = "weather-trading-bot"

# City name -> NWS station (must match cities.yaml)
CITY_STATIONS = {
    "New York": "KNYC",
    "Chicago": "KORD",
    "Los Angeles": "KLAX",
    "Miami": "KMIA",
    "Austin": "KAUS",
}


def fetch_actual_high_nws(station: str, target_date: date) -> float | None:
    """Fetch actual daily high from NWS hourly observations. Returns max temp in F."""
    start = datetime(target_date.year, target_date.month, target_date.day, 0, 0)
    end = start + timedelta(days=1)
    url = f"{NWS_API}/stations/{station}/observations"
    params = {
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 100,
    }
    try:
        resp = httpx.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        temps_f = []
        for feat in data.get("features", []):
            temp_c = feat["properties"]["temperature"]["value"]
            if temp_c is not None:
                temps_f.append(temp_c * 9 / 5 + 32)

        return max(temps_f) if temps_f else None
    except Exception as e:
        print(f"    NWS API error for {station} on {target_date}: {e}")
        return None


async def backfill_actuals():
    """Fill in missing actual_high_f values from NWS API."""
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row

    cursor = await db.execute(
        "SELECT * FROM forecast_actuals WHERE actual_high_f IS NULL"
    )
    rows = await cursor.fetchall()
    unfilled = [dict(r) for r in rows]

    if not unfilled:
        print("No unfilled actuals to backfill.")
        await db.close()
        return

    print(f"Backfilling {len(unfilled)} missing actuals...")
    filled = 0
    for record in unfilled:
        city = record["city"]
        forecast_date = date.fromisoformat(record["forecast_date"])

        # Only backfill dates that have passed
        if forecast_date >= date.today():
            print(f"  Skipping {city} {forecast_date} (future/today)")
            continue

        station = CITY_STATIONS.get(city)
        if not station:
            print(f"  Skipping {city}: no station mapping")
            continue

        actual = fetch_actual_high_nws(station, forecast_date)
        if actual is not None:
            await db.execute(
                "UPDATE forecast_actuals SET actual_high_f = ?, source = 'nws' WHERE id = ?",
                (actual, record["id"]),
            )
            await db.commit()
            filled += 1
            error = actual - record["model_mean_f"]
            print(f"  {city} {forecast_date}: model={record['model_mean_f']:.1f}F actual={actual:.1f}F error={error:+.1f}F")
        else:
            print(f"  {city} {forecast_date}: no NWS data available")

    print(f"Backfilled {filled}/{len(unfilled)} records.")
    await db.close()


async def export_hindcasts(output_dir: str = "./data/hindcasts/"):
    """Export forecast-vs-actual pairs as CSVs for bias model retraining."""
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row

    cursor = await db.execute(
        "SELECT * FROM forecast_actuals WHERE actual_high_f IS NOT NULL ORDER BY city, forecast_date"
    )
    rows = await cursor.fetchall()
    records = [dict(r) for r in rows]
    await db.close()

    if not records:
        print("No paired forecast-actual data to export. Run --backfill first.")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Group by city -> station
    by_city: dict[str, list] = {}
    for r in records:
        by_city.setdefault(r["city"], []).append(r)

    for city, city_records in by_city.items():
        station = CITY_STATIONS.get(city)
        if not station:
            continue

        path = out / f"{station.lower()}_hindcasts.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "ensemble_mean"])
            writer.writeheader()
            for r in city_records:
                writer.writerow({
                    "date": r["forecast_date"],
                    "ensemble_mean": r["model_mean_f"],
                })
        print(f"  {city} ({station}): exported {len(city_records)} pairs to {path}")

    print(f"\nTotal: {len(records)} paired records across {len(by_city)} cities.")
    print("Now retrain bias models: python -m scripts.train_bias_models --all-cities")


def main():
    parser = argparse.ArgumentParser(description="Backfill actuals and export hindcast pairs")
    parser.add_argument("--backfill", action="store_true", help="Backfill missing actuals from IEM")
    parser.add_argument("--export", action="store_true", help="Export paired CSVs for retraining")
    args = parser.parse_args()

    if not args.backfill and not args.export:
        parser.error("Specify --backfill, --export, or both")

    if args.backfill:
        asyncio.run(backfill_actuals())
    if args.export:
        asyncio.run(export_hindcasts())


if __name__ == "__main__":
    main()
