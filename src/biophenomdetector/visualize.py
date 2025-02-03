import datetime as dt
from enum import StrEnum
import re
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray as xr

from biophenomdetector import data_loading as data


class BiophysUnits(StrEnum):
    LAI = "$m^2/m^2$"
    CCC = "$\\mu g/cm^2$"
    CWC = "$g/cm^2$"


def normalize_image(img: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(img, (2, 98))
    return np.clip((img - p2) / (p98 - p2), 0, 1)


def generate_true_color_image(dataset: xr.Dataset) -> np.ndarray:
    rgb_bands = ["B4", "B3", "B2"]
    band_indices = [list(dataset.band.values).index(band) for band in rgb_bands]
    rgb_image = dataset.band_data.isel(time=0)[band_indices]
    return normalize_image(rgb_image.transpose("y", "x", "band").values)


def convert_date_to_string(
    date: Union[np.datetime64, pd.Timestamp, dt.date, dt.datetime],
    format: Optional[str] = "%Y-%m-%d",
) -> str:

    if isinstance(date, np.datetime64):
        date = date.astype(dt.datetime)
    elif isinstance(date, pd.Timestamp):
        date = date.to_pydatetime()

    return date.strftime(format)


def plot_satellite_image(dataset: xr.Dataset, **kwargs) -> None:
    image = generate_true_color_image(dataset)
    x_coords, y_coords = dataset.x.values, dataset.y.values

    epsg = dataset.rio.crs.to_epsg()
    date = convert_date_to_string(dataset.time.values.item())
    title = f"{date} - EPSG:{epsg}"

    fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    img = ax.imshow(
        image,
        extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
        **kwargs,
    )


def plot_biophys_result(
    data: xr.DataArray, cmap="rainbow", save_as: str = None, **kwargs
) -> None:
    image = plt.pcolormesh(data.x, data.y, data.values, cmap=cmap, **kwargs)

    epsg = data.rio.crs.to_epsg()
    date = convert_date_to_string(data.time.values.item())
    title = f"{data.name} - EPSG:{epsg} \n{date}"

    cbar = plt.colorbar(image, shrink=0.8)
    biophys_name = re.findall(r"\((.*?)\)", data.name)[0]
    unit = BiophysUnits[biophys_name].value
    cbar.set_label(f"{biophys_name} value in {unit}", rotation=270, labelpad=15)

    plt.title(title)
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")

    plt.tight_layout()
    if not save_as:
        plt.show()

    if save_as:
        plt.savefig(save_as)


def plot_anomalies(data: xr.DataArray, cmap="rainbow", **kwargs) -> None:
    image = plt.pcolormesh(data.x, data.y, data.values, cmap=cmap, **kwargs)

    epsg = data.rio.crs.to_epsg()
    pd_date = pd.to_datetime(data.time.values.item())
    date = convert_date_to_string(pd_date)
    title = f"Anomalies - EPSG:{epsg} \n{date}"

    cbar = plt.colorbar(image, shrink=0.8)
    cbar.set_label("Anomaly Value", rotation=270, labelpad=15)

    plt.title(title)
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")

    plt.tight_layout()
    plt.show()


def plot_anomaly_hist(data: xr.DataArray) -> None:
    values = data.values.flatten()

    pd_date = pd.to_datetime(data.time.values.item())
    date = convert_date_to_string(pd_date)
    title = f"Anomalies Histogram\n{date}"

    ax = sns.histplot(values)

    ax.set(xlabel="Anomaly values", title=title)
    plt.show()
