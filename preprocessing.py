from collections.abc import Sized
import re
from pathlib import Path

import numpy as np
import geopandas as gpd
from pyproj.crs.crs import CRS
import xarray as xr
import rasterio
import rioxarray
import pendulum


def read_raster_file(file_path: Path) -> xr.DataArray:
    """Read a raster file like a geotiff and return the content.

    Args:
        file_path:
            Path to the raster file.

    Returns:
         The contents of the raster file.
    """
    return rioxarray.open_rasterio(file_path)


def read_vector_file(file_path: Path) -> gpd.GeoDataFrame:
    """Read a vector file like a shapefile and return the content.

    file_path:
        Path to the vector file.

    Returns:
        The contents of the vector file.
    """
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


def crop_raster_data_to_shape_boundaries(
    raster_data: xr.DataArray, vector_data: gpd.GeoDataFrame
) -> xr.DataArray:
    """Crop the vector data to shape boundaries.

    Args:
        raster_data:
            The raster data to crop.
        vector_data:
            The vector data, where the boundaries can be extracted from.

    Returns:
        The cropped data in raster format.
    """
    extend = vector_data.bounds

    return raster_data.sel(
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
        # all_touched decides whether pixels that touch the border of a geometry are counted or not.
        all_touched=True,
        invert=True,
        out_shape=(raster_data.rio.height, raster_data.rio.width),
    )

    masked_data = raster_data.where(masked_data, np.nan)

    return masked_data


def create_grid_points(raster_data: xr.DataArray, interval: int = 10) -> np.ndarray:
    """Creates evenly-spaced points from the boundaries of a raster dataset.

    Args:
        raster_data:
            Raster dataset, only used to extract the boundaries.
        interval:
            Interval between the points.

    Returns:
        Array of all coordinates in the datasets boundaries.
    """

    img_extend = raster_data.rio.bounds()
    img_extend = [int(ext) for ext in img_extend]

    x_cor = np.arange(img_extend[0] + 5, img_extend[2] - 5 + 1, interval)
    y_cor = np.arange(img_extend[1] + 5, img_extend[3] - 5 + 1, interval)

    grid_x, grid_y = np.meshgrid(x_cor, y_cor)

    return np.vstack([grid_x.ravel(), grid_y.ravel()]).T


def create_points(data: np.ndarray, crs: CRS) -> gpd.GeoDataFrame:
    """Create geometry points from rasterized data.

    Args:
        data:
            Rasterized data array.
        crs:
            coordinate reference system.

    Returns:
        GeoDataFrame with points.
    """
    # wkt syntax is much faster than using geometry objects from shapely
    wkt_points = [f"Point ({data[i, 0]} {data[i, 1]})" for i in range(data.shape[0])]
    geo_series = gpd.GeoSeries.from_wkt(wkt_points)

    return gpd.GeoDataFrame(geometry=geo_series, crs=crs)


def crop_to_mask(pixels: gpd.GeoDataFrame, mask: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Crops geometry data to another geometry shape.

    Args:
        pixels:
            A geometry of points.
        mask:
            A mask of pixels, having NA for non-fitting values.

    Returns:
        The pixels that match the mask.
    """
    is_forest = ~np.isnan(mask.values.flatten())
    return pixels[is_forest]


def geom(data: gpd.GeoDataFrame) -> list[tuple[float]]:
    """Creates coordinates from a geometry. Analogue to the geom function from the terra package in R.

    Args:
        data:
            A geometry of points.

    Returns:
        A list of coordinates.
    """
    coords = data.geometry.apply(lambda geometry: geometry.coords[0])
    return coords.tolist()


def create_sequences(data: Sized, size: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Create sequences. Maybe removed in the future.

    Args:
        data:
            Any data with a length (len() can be applied).
        size:
            The size of the sequences.

    Returns:
        Two sequences as arrays.
    """
    from_vals = np.arange(1, len(data) + 1, size)
    to_vals = from_vals[1:] - 1

    return from_vals, to_vals


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

    dumimg: xr.DataArray = read_raster_file(sen2r_paths[0])
    dumimg2: xr.DataArray = crop_raster_data_to_shape_boundaries(dumimg, forest)
    dumimg3: xr.DataArray = mask_raster_with_vector(dumimg2, forest)

    xy: np.ndarray = create_grid_points(dumimg2)
    all_pix_pts: gpd.GeoDataFrame = create_points(xy, roi.crs)
    all_pix_pts_for: gpd.GeoDataFrame = crop_to_mask(all_pix_pts, dumimg3)
    xyfor: list[tuple[float]] = geom(all_pix_pts_for)


if __name__ == "__main__":
    main()
