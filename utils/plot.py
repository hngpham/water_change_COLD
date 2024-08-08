import os
import pickle
from xmlrpc.client import boolean
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np

from waterchange import utils

logger = utils.get_logger("DEBUG")


# %matplotlib inline
def plot_cold_coefs_from_all_coefs(
    input_data,
    sub_dir_out: str,
    coefs: List[Tuple],
    file_name: str,
    index: int = 0
) -> None:
    dim = len(coefs[0][0][1])
    print(dim)
    if dim != 8: 
        # Input is all_coefs, extract the index th coef of the coefs,
        # 200x200x[pos, list(8x7 coefs), num_obs, time]
        coefs = [[tuple_[1][index] for tuple_ in row] for row in coefs]
    else:
        # Input is first_coefs, extract the first coef of the coefs,
        # 200x200x[pos, 8x7 coefs, num_obs, time]
        coefs = [[tuple_[1] for tuple_ in row] for row in coefs]
    coefs = np.array(coefs)

    c_2_b456 = np.concatenate([coefs[:,:,2,4][:,:,np.newaxis], coefs[:,:,2,5][:,:,np.newaxis], coefs[:,:,2,6][:,:,np.newaxis]], axis=2)
    plt.figure(figsize=(20, 20))
    plt.imshow(c_2_b456)

    # Setting the major grid interval to 250 pixels
    plt.xticks(np.arange(0, coefs.shape[1], 250))
    plt.yticks(np.arange(0, coefs.shape[0], 250))
    
    # Adding major grid lines
    plt.grid(True, which='both', color='red', linestyle='--', linewidth=0.5)
    
    # Setting the minor grid interval to 50 pixels
    plt.gca().set_xticks(np.arange(-0.5, coefs.shape[1], 50), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, coefs.shape[0], 50), minor=True)
    
    # Adding minor grid lines
    plt.gca().grid(which='minor', color='red', linestyle=':', linewidth=0.5)
    file_path = utils.create_file_path(input_data.viz_dir, sub_dir_out, f"{file_name}.png")
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    logger.debug(f"Saved file at: {file_path}")
    plt.show()


def _plot_RGB_vs_water(
    separate: bool,
    date: str,
    regions: List[List[int]],
    index: int,
    rgb_image: List[List[np.ndarray]],
    water_mask: np.ndarray,
    output_folder: str,
    zoom: int = 1
) -> None:
    """
    Plots and saves side-by-side images of a lake's region, comparing the RGB image and a water mask for a given date.

    Parameters
    ----------
    date : str
        The date of the image.
    regions : list of tuples
        A list of coordinates representing bounding boxes of regions.
    index : int
        The index of the lake to be plotted.
    rgb_image : ndarray
        The RGB image array.
    water_mask : ndarray
        The water mask image array.
    output_folder : str
        The folder where the output image will be saved.
    zoom : int, optional
        The zoom factor to zoom in on the lake, by default 1.
    """
    zoom = (zoom - 1) / 2
    y1, y2, x1, x2 = regions[index]
    delta_y = zoom * (y2 - y1)
    delta_x = zoom * (x2 - x1)
    y1, y2, x1, x2 = int(y1 + delta_y), int(y2 - delta_y), int(x1 + delta_x), int(x2 - delta_x)

    if separate:
        # Create and save the RGB image
        fig_rgb, ax_rgb = plt.subplots(figsize=(10, 10))
        ax_rgb.imshow(rgb_image[y1:y2, x1:x2])
        ax_rgb.axis('off')  # Turn off axis ticks
        plt.savefig(f'{output_folder}/{index}_{date}_rgb.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig_rgb)

        # Create and save the water mask image
        fig_mask, ax_mask = plt.subplots(figsize=(10, 10))
        ax_mask.imshow(water_mask[y1:y2, x1:x2])
        ax_mask.axis('off')  # Turn off axis ticks
        plt.savefig(f'{output_folder}/{index}_{date}_mask.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig_mask)

    else:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(rgb_image[y1:y2, x1:x2])
        ax[0].axis('off')  # Turn off axis ticks
        ax[1].imshow(water_mask[y1:y2, x1:x2])
        ax[1].axis('off')  # Turn off axis ticks
        
        plt.suptitle(f'Image Date: {date}')
        plt.savefig(f'{output_folder}/{index}_{date}.png', bbox_inches='tight', pad_inches=0)
        plt.close()


def plot_RGB_vs_water_years(separate, input_data, indices, years, zoom: int = 1):
    from waterchange.datasets import load_landsat_image

    for year in years:
        # Extract all subfolder of 1984
        dirs = utils.get_all_subdirectories(f"{input_data.landsat_dir}/{year}")
        output_folder = f"{input_data.viz_dir}/RGB_vs_water_{year}"
        os.makedirs(output_folder, exist_ok=True)
        for path in dirs:
            # Plot the image and same the water mask for each lake
            img, date = load_landsat_image(path,[4,3,2,5])
            # natural color
            rgb_image = utils.bands_to_rgb(img, 4, 3, 2, alpha=2.)
            # Calculating two indices
            # ndvi = utils.normalized_difference(img, 5, 4)
            mndwi = utils.normalized_difference(img, 3, 5)
            water_mask = mndwi > 0.0
            for index in indices:
                _plot_RGB_vs_water(separate,
                                   date,
                                   input_data.regions,
                                   index, rgb_image,
                                   water_mask,
                                   output_folder,
                                   zoom)
