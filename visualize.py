import re

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import data_loading as data


def normalize_image(img: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(img, (2, 98))
    return np.clip((img - p2) / (p98 - p2), 0, 1)


def generate_true_color_image(dataset: xr.Dataset) -> np.ndarray:
    rgb_bands = ["B4", "B3", "B2"]
    band_indices = [list(dataset.band.values).index(band) for band in rgb_bands]
    rgb_image = dataset.band_data.isel(time=0)[band_indices]
    return normalize_image(rgb_image.transpose("y", "x", "band").values)


def plot_satellite_image(dataset: xr.Dataset, crs: data.CRS, **kwargs) -> None:
    image = generate_true_color_image(dataset)
    x_coords, y_coords = dataset.x.values, dataset.y.values

    title = f"{crs.name.replace("_", " ")} ({crs.ogc_string()})"

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
    data: xr.DataArray, crs: data.CRS, cmap="rainbow", **kwargs
) -> None:
    image = plt.pcolormesh(data.x, data.y, data.values, cmap=cmap, **kwargs)

    title = f"{data.name} - {crs.name.replace("_", " ")} ({crs.ogc_string()})"

    cbar = plt.colorbar(image, shrink=0.8)
    biophys_name_short = re.findall(r"\((.*?)\)", data.name)[0]
    cbar.set_label(f"{biophys_name_short} Value", rotation=270, labelpad=15)

    plt.title(title)
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")

    plt.tight_layout()
    plt.show()
