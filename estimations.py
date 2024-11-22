from enum import StrEnum
import datetime as dt
from pathlib import Path
from typing import Union

import pandas as pd
import xarray as xr
import shapely
from satellitetools.biophys.biophys import run_snap_biophys

import data_loading as data
import preprocessing as pp

spectral_metadata = {
    "B1": {"description": "Ultra blue band", "units": "nm"},
    "B2": {"description": "Blue band", "units": "nm"},
    "B3": {"description": "Green band", "units": "nm"},
    "B4": {"description": "Green band", "units": "nm"},
    "B5": {"description": "Red band", "units": "nm"},
    "B6": {"description": "Red Edge 1", "units": "nm"},
    "B7": {"description": "Red Edge 2", "units": "nm"},
    "B8": {"description": "NIR", "units": "nm"},
    "B8A": {"description": "NIR Narrow", "units": "nm"},
    "B9": {"description": "Water Vapour", "units": "nm"},
    "B11": {"description": "SWIR 1", "units": "nm"},
    "B12": {"description": "SWIR 2", "units": "nm"},
}

angle_metadata = {
    "sun_azimuth": {"description": "Sun azimuth angle", "units": "degrees"},
    "sun_zenith": {"description": "Sun elevation angle", "units": "degrees"},
    "view_azimuth": {"description": "Viewing azimuth angle", "units": "degrees"},
    "view_zenith": {"description": "Viewing elevation angle", "units": "degrees"},
}
spectral_band_names = list(spectral_metadata.keys())
angle_band_names = list(angle_metadata.keys())
all_band_names = list(spectral_band_names) + list(angle_band_names)


def create_dataset(image: data.Sentinel2Image) -> xr.Dataset:

    data_array = pp.read_raster_file(image["path"], masked=True)
    data_array = data_array.assign_coords(band=("band", all_band_names))
    data_array = data_array.expand_dims(time=[image["date"]])

    spectral_data = data_array.sel(band=spectral_band_names)
    angle_data = data_array.sel(band=angle_band_names)

    dataset = spectral_data.to_dataset(name="band_data")

    for angle in angle_band_names:
        dataset = dataset.assign({angle: angle_data.sel(band=angle)})

    dataset = dataset.rename({"y": "temp_y", "x": "y"}).rename({"temp_y": "x"})
    dataset["band_data"] = dataset["band_data"].transpose("time", "band", "y", "x")

    for angle in angle_band_names:
        dataset[angle] = dataset[angle].transpose("x", "y", "time")

    for band in spectral_band_names:
        dataset["band_data"].attrs.setdefault(band, {}).update(
            spectral_metadata.get(band, {})
        )

    for angle in angle_band_names:
        dataset[angle].attrs.update(angle_metadata.get(angle, {}))

    return dataset


class BiophysicalVariable(StrEnum):
    LAI = "LAI"
    CCC = "LAI_Cab"
    CWC = "LAI_Cw"


def estimate_lai(dataset: xr.Dataset) -> xr.DataArray:
    return run_snap_biophys(dataset, BiophysicalVariable.LAI)


def estimate_ccc(dataset: xr.Dataset) -> xr.DataArray:
    return run_snap_biophys(dataset, BiophysicalVariable.CCC)


def estimate_cwc(dataset: xr.Dataset) -> xr.DataArray:
    return run_snap_biophys(dataset, BiophysicalVariable.CWC)

def estimate_all_vars(dataset: xr.Dataset) -> xr.Dataset:
    for var in BiophysicalVariable:
            dataset = run_snap_biophys(dataset, var)

    return dataset
