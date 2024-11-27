import preprocessing as pp

import datetime as dt
from pathlib import Path
import re

import numpy as np
import pandas as pd
import xarray as xr


def create_time_series(paths: list[Path]) -> xr.DataArray:
    data_array = [pp.read_raster_file(path)[0] for path in paths]
    datetimes = [re.search(r"\d{8}", path.stem).group() for path in paths]

    data_stack = np.stack(data_array, axis=0)

    return xr.DataArray(
        data_stack,
        dims=["time", "y", "x"],
        coords={"time": pd.to_datetime(datetimes)},
        name="raster_data",
    )


def fill_gaps(
    data: xr.DataArray, timedelta: str = "7D", method: str = "linear"
) -> xr.DataArray:
    filled = data.resample(time=timedelta).mean()

    return filled.interpolate_na(dim="time", method=method, fill_value="extrapolate")


def compute_standardized_anomaly(
    data: xr.DataArray, average: xr.DataArray, std: xr.DataArray, time: dt.datetime
) -> xr.DataArray:
    data = data.sel(time=time)
    return (data - average) / std
