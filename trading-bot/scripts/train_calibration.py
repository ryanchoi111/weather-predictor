"""Train Platt scaling calibration models from settled contract outcomes.

Usage:
    python -m scripts.train_calibration              # train global + per-city models
    python -m scripts.train_calibration --report      # print calibration report only
    python -m scripts.train_calibration --min-samples 20  # lower threshold for training

Requires settled contracts in the database (populated automatically each cycle).
"""

import argparse
import asyncio
from pathlib import Path

import aiosqlite

from src.calibration.feedback import (
    build_calibration_model,
    save_calibration_model,
    generate_calibration_report,
)

DB_PATH = Path(__file__).parent.parent / "data" / "trading.db"

CITIES = ["New York", "Chicago", "Los Angeles", "Miami", "Austin"]


async def load_settled_contracts() -> list[dict]:
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    cursor = await db.execute(
        "SELECT * FROM settled_contracts WHERE model_prob IS NOT NULL ORDER BY settled_at"
    )
    rows = await cursor.fetchall()
    await db.close()
    return [dict(r) for r in rows]


async def run_report():
    data = await load_settled_contracts()
    if not data:
        print("No settled contracts with model predictions found.")
        print("Run the bot for a few days to accumulate data.")
        return

    report = generate_calibration_report(data)
    print(f"\n=== Calibration Report ===")
    print(f"Total contracts: {report['total_contracts']}")
    print(f"Traded contracts: {report['total_traded']}")
    print(f"Total P&L: ${report['total_pnl_cents'] / 100:.2f}")
    print(f"Overall Brier Score: {report['brier_score']}")
    print(f"\nReliability by probability bin:")
    print(f"{'Bin':<12} {'N':>5} {'Predicted':>10} {'Actual':>8} {'Gap':>8}")
    print("-" * 45)
    for b in report["bins"]:
        marker = " !" if abs(b["gap"]) > 0.1 else ""
        print(f"{b['range']:<12} {b['n']:>5} {b['predicted']:>10.1%} {b['actual']:>8.1%} {b['gap']:>+8.1%}{marker}")

    # Per-city breakdown
    for city in CITIES:
        city_data = [d for d in data if d["city"] == city]
        if not city_data:
            continue
        city_report = generate_calibration_report(city_data)
        traded = [d for d in city_data if d.get("traded")]
        pnl = sum(d.get("pnl_cents", 0) or 0 for d in traded)
        print(f"\n  {city}: {len(city_data)} contracts, "
              f"Brier={city_report['brier_score']}, "
              f"{len(traded)} traded, P&L=${pnl / 100:.2f}")


async def run_training(min_samples: int = 30):
    data = await load_settled_contracts()
    if not data:
        print("No settled contracts with model predictions found.")
        return

    print(f"Training on {len(data)} settled contracts...\n")

    # Global model
    global_model = build_calibration_model(data, city=None, min_samples=min_samples)
    if global_model:
        save_calibration_model(global_model)
        print(f"Global: Brier {global_model.brier_before:.4f} -> {global_model.brier_after:.4f} "
              f"({global_model.n_samples} samples)")
    else:
        print(f"Global: not enough data (need {min_samples}+)")

    # Per-city models
    for city in CITIES:
        model = build_calibration_model(data, city=city, min_samples=min_samples)
        if model:
            save_calibration_model(model)
            print(f"{city}: Brier {model.brier_before:.4f} -> {model.brier_after:.4f} "
                  f"({model.n_samples} samples)")
        else:
            city_n = len([d for d in data if d["city"] == city])
            print(f"{city}: {city_n} samples (need {min_samples}+), using global fallback")


def main():
    parser = argparse.ArgumentParser(description="Train calibration models from settled contracts")
    parser.add_argument("--report", action="store_true", help="Print calibration report only")
    parser.add_argument("--min-samples", type=int, default=30, help="Min samples to train")
    args = parser.parse_args()

    if args.report:
        asyncio.run(run_report())
    else:
        asyncio.run(run_training(args.min_samples))
        print("\nRun with --report to see full calibration breakdown.")


if __name__ == "__main__":
    main()
