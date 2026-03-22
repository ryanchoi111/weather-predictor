import httpx

from .data.logger import get_logger

log = get_logger("nws")


async def get_current_temp_f(station: str) -> float | None:
    """Fetch current observed temperature from NWS API. Returns temp in Fahrenheit or None on failure."""
    url = f"https://api.weather.gov/stations/{station}/observations/latest"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers={"User-Agent": "weather-trading-bot"})
            resp.raise_for_status()
            data = resp.json()
            temp_c = data["properties"]["temperature"]["value"]
            if temp_c is None:
                return None
            return temp_c * 9 / 5 + 32
    except Exception as e:
        log.warning("nws_fetch_failed", station=station, error=str(e))
        return None
