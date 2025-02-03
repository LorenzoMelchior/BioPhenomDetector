from biophenomdetector import preprocessing as pp
from biophenomdetector import estimations as em

import datetime as dt
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import xarray as xr


def create_time_series(
    images: dict, biophysical_var: em.BiophysicalVariable
) -> xr.DataArray:
    images = sorted(images, key=lambda x: x["date"])
    datasets = [em.create_dataset(image) for image in images]
    match biophysical_var:
        case em.BiophysicalVariable.LAI:
            estimations = [em.estimate_lai(ds) for ds in datasets]
        case em.BiophysicalVariable.CCC:
            estimations = [em.estimate_ccc(ds) for ds in datasets]
        case em.BiophysicalVariable.CWC:
            estimations = [em.estimate_cwc(ds) for ds in datasets]

    datetimes = [item.time.values.item() for item in estimations]
    data_stack = np.stack([item.values for item in estimations], axis=0)

    return xr.DataArray(
        data_stack,
        dims=["time", "y", "x"],
        coords={
            "time": pd.to_datetime(datetimes),
            "y": estimations[0].y,
            "x": estimations[0].x,
        },
        name=f"{biophysical_var.name}",
    ).rio.write_crs(datasets[0].rio.crs)


def fill_gaps(
    data: xr.DataArray, timedelta: str = "7D", method: str = "linear"
) -> xr.DataArray:
    filled = data.resample(time=timedelta).mean()

    return filled.interpolate_na(dim="time", method=method, fill_value="extrapolate")


def fill_gaps_and_smooth_data(
    data_array: xr.DataArray, frequency: str = dt.timedelta(days=1)
):

    window_length = 21
    polyorder = 3

    start_date = data_array.time.min().values
    end_date = data_array.time.max().values
    full_date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)

    filled_array = data_array.reindex(time=full_date_range)

    for y in range(filled_array.y.size):
        for x in range(filled_array.x.size):
            pixel_data = filled_array[:, y, x].values

            mask = np.isnan(pixel_data)
            non_nan_indices = np.flatnonzero(~mask)
            if len(non_nan_indices) > 1:
                pixel_data = np.interp(
                    np.arange(len(pixel_data)), non_nan_indices, pixel_data[~mask]
                )

                smoothed_data = savgol_filter(pixel_data, window_length, polyorder)
                filled_array[:, y, x] = smoothed_data
            elif len(non_nan_indices) == 1:
                filled_array[:, y, x] = pixel_data[non_nan_indices[0]]

    return filled_array


def compute_sda_for_dayofyear(data: xr.DataArray, date: dt.date) -> xr.DataArray:
    """Computes the multi-year standardized anomaly for a given date.
    The given date will filter the dataset to only include the same day of the year
    over multiple years (e.g. 1st of January for each year).

    Args:
        data:
            A full interpolated dataset with values of a biophysical variable
            estimation over multiple years.
        date:
            The date to be analyzed. Will also be used to filter the dataset.

    Returns:
        An array of computed anomalies.
    """
    # needed, as they would not fully match otherwise
    np_date = np.datetime64(date)
    data_on_date = data.sel(time=np_date)

    # e.g. 1 for first of January
    day_of_year = pd.to_datetime(date).dayofyear

    # all samples for the given day of year
    filtered = data.groupby("time.dayofyear")[day_of_year]

    mean = filtered.mean(dim="time")
    std = filtered.std(dim="time", skipna=False)

    return (data_on_date - mean) / std
