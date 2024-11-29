import preprocessing as pp
import estimations as em

import datetime as dt
from pathlib import Path
import re

import numpy as np
import pandas as pd
import xarray as xr


# def create_time_series(paths: list[Path]) -> xr.DataArray:
#    data_array = [pp.read_raster_file(path)[0] for path in paths]
#    datetimes = [re.search(r"\d{8}", path.stem).group() for path in paths]

#    data_stack = np.stack(data_array, axis=0)

#    return xr.DataArray(
#        data_stack,
#        dims=["time", "y", "x"],
#        coords={"time": pd.to_datetime(datetimes)},
#        name="raster_data",
#    )


def create_time_series(
    images: dict, biophysical_var: em.BiophysicalVariable
) -> xr.DataArray:
    images = sorted(images, key=lambda x: x["date"])
    datasets = [em.create_dataset(image) for image in images]
    if biophysical_var == em.BiophysicalVariable.LAI:
        estimations = [em.estimate_lai(ds) for ds in datasets]
    elif biophysical_var == em.BiophysicalVariable.CCC:
        estimations = [em.estimate_ccc(ds) for ds in datasets]
    elif biophysical_var == em.BiophysicalVariable.CWC:
        estimations = [em.estimate_cwc(ds) for ds in datasets]

    datetimes = [item.time.values.item() for item in estimations]
    data_stack = np.stack([item.values for item in estimations], axis=0)

    return xr.DataArray(
        data_stack,
        dims=["time", "y", "x"],
        coords={"time": pd.to_datetime(datetimes)},
        name=f"{biophysical_var.name}",
    )


def fill_gaps(
    data: xr.DataArray, timedelta: str = "7D", method: str = "linear"
) -> xr.DataArray:
    filled = data.resample(time=timedelta).mean()

    return filled.interpolate_na(dim="time", method=method, fill_value="extrapolate")


def compute_standardized_anomaly(data: xr.DataArray, time: dt.datetime) -> xr.DataArray:

    mean = data.mean(dim="time")
    std = data.std(dim="time")
    data = data.sel(time=time, method="nearest")

    return (data - mean) / std
