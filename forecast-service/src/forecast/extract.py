import numpy as np


def extract_point_temperature(
    output,
    latitude: float,
    longitude: float,
    lead_step: int = -1,
) -> list[float]:
    """Extract t2m at a point from ensemble output, convert K to F.

    Args:
        output: xarray Dataset from ensemble run
        latitude: city latitude
        longitude: city longitude (-180..180 convention, converted internally to 0..360)
        lead_step: which forecast step to use (-1 = last)

    Returns:
        List of temperatures in Fahrenheit, one per ensemble member.
    """
    lon_360 = longitude % 360

    lats = output.coords["lat"].values
    lons = output.coords["lon"].values
    lat_idx = int(np.argmin(np.abs(lats - latitude)))
    lon_idx = int(np.argmin(np.abs(lons - lon_360)))

    temps_k = output["t2m"].values[:, lead_step, lat_idx, lon_idx]
    temps_f = (temps_k - 273.15) * 9 / 5 + 32

    return temps_f.tolist()
