import datetime as dt
import getpass
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Union

import pyproj
import shapely
import xarray as xr

from tqdm import tqdm

from sentinelhub import (
    BBox,
    CRS,
    DataCollection,
    MimeType,
    SentinelHubCatalog,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)

MIN_FILE_SIZE = 1024 * 1024
Sentinel2Image = dict[str, Union[str, dt.date, Path]]

relevant_bands = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
    "sunAzimuthAngles",
    "sunZenithAngles",
    "viewAzimuthMean",
    "viewZenithMean",
]
# relevant_bands = ["B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]


def create_configuration(
    client_id: Optional[str] = None, client_secret: Optional[str] = None
) -> SHConfig:
    config = SHConfig()
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"

    if client_id and client_secret:
        config.sh_client_id = client_id
        config.sh_client_secret = client_secret
    elif not config.sh_client_id and not config.sh_client_secret:
        config.sh_client_id = getpass.getpass("Enter your SentinelHub client id")
        config.sh_client_secret = getpass.getpass(
            "Enter your SentinelHub client secret"
        )

    return config


def save_as_tiff(data: xr.DataArray, name: str) -> None:
    data.rio.to_raster(name)


def parse_date(time: str) -> dt.date:
    """Parses a datetime in a given pattern and converts it
    to a datetime object, reducing the accuracy to day (year, month, day).
    Input:

    Args:
        time:
            A time string with (Year, Month, Day, Minute, Hour, Seconds, Microseconds)
            in the correct pattern.

    Returns:
        The correspoinding date (Year, Month, Day).
    """

    pattern = "%Y-%m-%dT%H:%M:%S.%fZ"
    parsed_time: dt.datetime = dt.datetime.strptime(time, pattern)
    parsed_date: dt.date = parsed_time.date()

    return parsed_date


def generate_name(ids: list[str]) -> str:
    """Generates name of the sentinel-2 recording based on its ids.
    As multiple recordings may be mosaiced into one image, differences
    are concatenated, e.g. if images from S2A and S2B are being used,
    the resulting filename will start with S2AS2B.

    Args:
        ids:
            A list of >=1 ids to be used for one filename.

    Returns:
        Filename with satellite name, preprocessing level, recording date,
        processing baseline and relative orbit.
    """

    DELIMITERS = "_."

    substrings = list(map(lambda text: re.split(rf"[{DELIMITERS}]", text), ids))

    extract_element = lambda index: "".join(
        list({substring[index] for substring in substrings})
    )

    satellite_name = extract_element(0)
    preprocessing_level = extract_element(1)
    date = extract_element(2)
    processing_baseline = extract_element(3)
    relative_orbit = extract_element(4)

    return "_".join(
        [satellite_name, preprocessing_level, date, processing_baseline, relative_orbit]
    )


def get_recordings(query_result: list[dict[str, str]]) -> dict[str, dt.date]:

    recordings = [
        {"id": res["id"], "date": parse_date(res["properties"]["datetime"])}
        for res in query_result
    ]

    unique_dates = {item["date"] for item in recordings}

    result = []

    for date in unique_dates:
        ids = [recording["id"] for recording in recordings if recording["date"] == date]
        name = generate_name(ids)

        result.append({"name": name, "date": date})

    return result


def query_copernicushub(
    config: SHConfig, bbox: BBox, timeframe: tuple[dt.date]
) -> list[Sentinel2Image]:

    catalog = SentinelHubCatalog(config=config)
    search_results = list(
        catalog.search(
            DataCollection.SENTINEL2_L2A,
            bbox=bbox,
            time=timeframe,
            fields={"include": ["id", "properties.datetime"], "exclude": []},
        )
    )

    return get_recordings(search_results)


def generate_evalscript(relevant_bands: list[str]) -> str:
    return f"""
//VERSION=3
function setup() {{
    return {{
        input: {str(relevant_bands)},
        output: {{
            bands: {len(relevant_bands)},
            sampleType: "FLOAT32"
        }},
        processing: {{
            upsampling: "BILINEAR"
        }}
    }};
}}

function evaluatePixel(sample) {{
    return [
        {', '.join([f'sample.{band}' for band in relevant_bands])}
    ];
}}
"""


def download_single_satellite_image(
    evalscript: str,
    config: SHConfig,
    bbox: BBox,
    time_interval: tuple[dt.date],
    resulting_file_path: Path,
    temporary_folder: Path = Path("tmp"),
    resolution: int = 60,
    max_cloud_coverage: float = 0.2,
) -> None:

    temporary_folder.mkdir(exist_ok=True)

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A.define_from(
                    "s2", service_url=config.sh_base_url
                ),
                time_interval=sorted(
                    tuple(map(lambda x: x.strftime("%Y-%m-%d"), time_interval))
                ),
                maxcc=max_cloud_coverage,
                mosaicking_order="leastCC",
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=bbox_to_dimensions(bbox, resolution=resolution),
        config=config,
        data_folder=str(temporary_folder.absolute()),
    )

    request.get_data(save_data=True)
    tmp_file_path = Path(temporary_folder) / Path(request.get_filename_list()[0])

    file_size = os.path.getsize(tmp_file_path)
    if file_size < MIN_FILE_SIZE:
        raise RuntimeError("File too small")

    tmp_file_path.rename(resulting_file_path)


def use_local_files(
    directory: Path, time_range: tuple[dt.date]
) -> list[Sentinel2Image]:

    file_pattern = re.compile(r"S2[A|B|C|D]_MSIL2A_(\d{8}).*\.tiff?")

    matching_files = []
    dates_found = []

    for file in directory.glob("*.tif*"):
        match = file_pattern.search(file.name)
        if match:
            file_date = dt.datetime.strptime(match.group(1), "%Y%m%d").date()

            if (
                time_range[0] <= file_date <= time_range[1]
                and not file_date in dates_found
            ):
                matching_files.append(
                    {"name": file.stem, "date": file_date, "path": file}
                )
                dates_found.append(file_date)

    return sorted(matching_files, key=lambda x: x["date"])


def load_satellite_images(
    aoi: dict[str, Union[shapely.box, pyproj.crs.crs.CRS]],
    time_range: tuple[dt.date],
    file_path: Path,
    config: SHConfig = None,
    download=True,
    show_progress: bool = False,
    tmp_dir: Path = Path("tmp"),
) -> list[Sentinel2Image]:

    if not download:
        return use_local_files(file_path, time_range)

    if not config:
        config = create_configuration()

    bbox = BBox(aoi["bbox"], aoi["crs"])

    available_recordings = query_copernicushub(config, bbox, time_range)

    evalscript = generate_evalscript(relevant_bands)

    file_path.mkdir(exist_ok=True)
    tmp_exists = tmp_dir.is_dir()
    tmp_dir.mkdir(exist_ok=True)

    for item in available_recordings:
        item["path"] = file_path / f"{item['name']}.tif"

    downloaded = []
    for item in tqdm(available_recordings, unit="image", disable=not show_progress):

        if not item["path"].is_file():
            delta = dt.timedelta(days=1)
            time_interval = (item["date"] - delta, item["date"] + delta)
            try:
                download_single_satellite_image(
                    evalscript, config, bbox, time_interval, item["path"]
                )
            except:
                continue
        downloaded.append(item)

    if not tmp_exists:
        shutil.rmtree(tmp_dir)

    return sorted(downloaded, key=lambda x: x["date"])
