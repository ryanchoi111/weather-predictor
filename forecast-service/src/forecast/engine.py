import numpy as np


async def run_ensemble(
    model_name: str,
    nensemble: int = 30,
    nsteps: int = 4,
    batch_size: int = 8,
) -> np.ndarray:
    """Run ensemble forecast, returns zarr-backed output array.

    Returns xarray Dataset with dimensions (ensemble, time, lat, lon).
    """
    from earth2studio.run import ensemble
    from earth2studio.perturbation import SphericalGaussian
    from earth2studio.data import GFS
    from earth2studio.io import ZarrBackend
    from .models import load_model

    model = load_model(model_name)
    perturbation = SphericalGaussian(noise_amplitude=0.15)
    data_source = GFS()
    io_backend = ZarrBackend()

    output = ensemble(
        model=model,
        perturbation=perturbation,
        data=data_source,
        io=io_backend,
        nensemble=nensemble,
        nsteps=nsteps,
        batch_size=batch_size,
        output_coords={"variable": ["t2m"]},
    )

    return output
