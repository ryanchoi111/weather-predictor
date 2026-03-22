import csv
import io
from dataclasses import dataclass
from datetime import date, datetime

import httpx

from ..data.logger import get_logger

log = get_logger("metar_client")


@dataclass
class METARObservation:
    station: str
    time: datetime
    temp_f: float
    is_valid: bool


async def fetch_todays_observations(
    icao_code: str, tz: str = "America/New_York"
) -> list[METARObservation]:
    """Fetch today's hourly METAR observations from IEM ASOS archive."""
    today = date.today()
    station = icao_code.lstrip("K")
    params = {
        "station": station,
        "data": "tmpf",
        "year1": today.year, "month1": today.month, "day1": today.day,
        "year2": today.year, "month2": today.month, "day2": today.day,
        "tz": tz,
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": "3",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
                params=params,
            )
            resp.raise_for_status()

        observations = []
        reader = csv.DictReader(io.StringIO(resp.text))
        for row in reader:
            tmpf = row.get("tmpf", "M")
            if tmpf == "M" or not tmpf:
                continue
            try:
                obs = METARObservation(
                    station=icao_code,
                    time=datetime.strptime(row["valid"], "%Y-%m-%d %H:%M"),
                    temp_f=float(tmpf),
                    is_valid=True,
                )
                observations.append(obs)
            except (ValueError, KeyError):
                continue

        return sorted(observations, key=lambda o: o.time)
    except Exception as e:
        log.warning("metar_fetch_failed", station=icao_code, error=str(e))
        return []


async def fetch_latest_observation(icao_code: str) -> METARObservation | None:
    """Fetch latest METAR from NOAA Aviation Weather API."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"https://aviationweather.gov/api/data/metar?ids={icao_code}&format=json",
            )
            resp.raise_for_status()
            data = resp.json()

        if not data:
            return None

        entry = data[0]
        temp_c = entry.get("temp")
        if temp_c is None:
            return None

        return METARObservation(
            station=icao_code,
            time=datetime.fromisoformat(entry["reportTime"].replace("Z", "+00:00")),
            temp_f=temp_c * 9 / 5 + 32,
            is_valid=True,
        )
    except Exception as e:
        log.warning("metar_latest_failed", station=icao_code, error=str(e))
        return None
