import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def normalize_image(img: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(img, (2, 98))
    return np.clip((img - p2) / (p98 - p2), 0, 1)

def generate_satellite_image(dataset: xr.Dataset) -> np.ndarray:
    rgb_bands = ['B4', 'B3', 'B2']
    band_indices = [list(dataset.band.values).index(band) for band in rgb_bands]
    rgb_image = dataset.band_data.isel(time=0)[band_indices]
    return normalize_image(rgb_image.transpose('y', 'x', 'band').values)

def plot_satellite_image(dataset: xr.Dataset, **kwargs) -> None:
    image = generate_satellite_image(dataset)
    x_coords, y_coords = dataset.x.values, dataset.y.values
    plt.imshow(image, extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],**kwargs)
