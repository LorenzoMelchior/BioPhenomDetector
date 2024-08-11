import re
from pathlib import Path

import numpy as np
import geopandas as gpd
import xarray as xr
import rasterio
import rioxarray
import pendulum


def read_raster_file(file_path: Path) -> xr.DataArray:
    return rioxarray.open_rasterio(file_path)


def read_vector_file(file_path: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(file_path)


def load_filtered_paths(
    directory: Path,
    start_date: pendulum.DateTime,
    end_date: pendulum.DateTime,
    file_extension: str,
) -> list[Path]:
    """Loads all file-paths in a given time range and a specific file-extension.

    Args:
        directory:
            A root directory to search for file-paths.
        start_date:
            The start date of the time range.
        end_date:
            The end date of the time range.
        file_extension:
            The file extension of the file-paths without the leading dot.

    Returns:
        A list of paths for matching files.
    """
    filtered_paths = []

    for file in directory.rglob(f"*.{file_extension}"):
        date_match = re.search(r"\d{8}", file.stem)
        if date_match:
            file_date = pendulum.parse(date_match.group())
            if start_date <= file_date <= end_date:
                filtered_paths.append(file)

    return sorted(filtered_paths)


def crop_vector_data_to_shape_boundaries(
    vector_data: xr.DataArray, shape: gpd.GeoDataFrame
) -> xr.DataArray:
    extend = shape.bounds

    return vector_data.sel(
        x=slice(extend.minx.values[0], extend.maxx.values[0]),
        y=slice(
            extend.maxy.values[0],  # note that y must be sliced from max to min
            extend.miny.values[0],
        ),
    )


def mask_raster_with_vector(
    raster_data: xr.DataArray,
    vector_mask: gpd.GeoDataFrame,
) -> xr.DataArray:
    """Masks a dataset in raster format using a vector mask.

    Args:
        raster_data:
            The raster-dataset to be masked.
        vector_mask:
            A vector-dataset, where the mask can be taken from.

    Returns:
        The masked raster-dataset.
    """
    data = raster_data.rio.write_crs(raster_data.rio.crs)
    transform = data.rio.transform()

    masked_data = rasterio.features.geometry_mask(
        [vector_mask["geometry"][0]],
        transform=transform,
        invert=True,
        out_shape=(raster_data.rio.height, raster_data.rio.width),
    )

    masked_data = raster_data.where(masked_data, np.nan)

    return masked_data


def create_data_array(raster_data: xr.DataArray) -> np.ndarray:

    img_extend = raster_data.rio.bounds()
    img_extend = [int(ext) for ext in img_extend]

    x_cor = np.arange(img_extend[0] + 5, img_extend[2] - 5 + 1, 10)
    y_cor = np.arange(img_extend[1] + 5, img_extend[3] - 5 + 1, 10)

    grid_x, grid_y = np.meshgrid(x_cor, y_cor)

    return np.vstack([grid_x.ravel(), grid_y.ravel()]).T


def main():

    roi = read_vector_file(
        Path("data/1_Bea_Ticino/ROI_30x30_rilievo_floristico_UTM_finale_CINZIA.shp")
    )
    forest = read_vector_file(Path("data/1_Bea_Ticino/border_forest_final2.shp"))

    sen2r_paths = load_filtered_paths(
        directory=Path("/Users/lorenzo/MA-local/data/CWC_rename"),
        start_date=pendulum.parse("2017-01-01"),
        end_date=pendulum.parse("2023-12-31"),
        file_extension="tif",
    )

    dumimg = read_raster_file(sen2r_paths[0])
    dumimg2 = crop_vector_data_to_shape_boundaries(dumimg, forest)
    dumimg3 = mask_raster_with_vector(dumimg2, forest)

    print(create_data_array(dumimg2))


if __name__ == "__main__":
    main()
