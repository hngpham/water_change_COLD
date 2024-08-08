import os

import glob
import rasterio
import numpy as np
import pickle

from typing import Tuple, Dict, List, Union

from waterchange import utils

logger = utils.get_logger("DEBUG")


def load_landsat_image(
    img_folder: str,
    bands: List[int],
) -> Tuple[Dict[int, np.ndarray], Union[str, None]]:
    # 1_indexed
    image = {}
    date = None

    for band in bands:
        try:
            # Locate the correct file for the given band
            # 1-indexed
            file_pattern = f'{img_folder}/*_SR_B{band}.TIF'
            file_list = glob.glob(file_pattern)

            if not file_list:
                raise FileNotFoundError(f"No file found for band {band} in folder {img_folder}")

            file = file_list[0]
            # Extract date from the filename
            date = file.split('/')[-1].split('_')[3]
            logger.debug(f'Opening file {file}')

            # Open the file and read the band data
            with rasterio.open(file) as ds:
                image[band] = ds.read(1)

        except Exception as e:
            logger.error(f"Error loading band {band}: {e}")

    return image, date


def export_average_water_index_per_year(
    input_data, sub_dir: str,
    start_year: int,
    end_year: int,
    region_id: int,
    index_threshold: float = 0.0,
    black_threshold: int = 10,
    band1: int = 2,
    band2: int = 5
) -> None:
    """
    Export the average water index per year for a specified region.

    Parameters:
    start_year : int
        The start year for the range of data.
    end_year : int
        The end year for the range of data.
    region_id : int
        The region ID to be processed.
    index_threshold : float, optional
        Threshold for the water index to classify water (default is 0.0).
    black_threshold : int, optional
        Percentage threshold for black pixels to skip images (default is 10).

    Returns:
    None
    """
    logger.debug(f"Processing region {region_id} at {input_data.regions[region_id]}.")
    for year in range(start_year, end_year + 1):
        dirs = utils.get_all_subdirectories(f"{input_data.landsat_dir}/{year}")
        data = {}
        masks = []

        for directory in dirs:
            img, date = load_landsat_image(directory, [4, 3, 2, 5]) # 1-indexed
            # Natural color image
            rgb_image = utils.bands_to_rgb(img, 4, 3, 2, alpha=2.0)
            # Calculating MNDWI
            mndwi = utils.normalized_difference(img, band1, band2) # 1-indexed
            water_mask = np.array(mndwi > index_threshold, dtype=int)

            # Extract region coordinates
            y1, y2, x1, x2 = input_data.regions[region_id]
            black = utils.calculate_black_pixel_percentage(rgb_image[y1:y2, x1:x2])

            if black > black_threshold:
                logger.debug(f"Black pixel percentage: {black:.2f}%")
                logger.debug(f"Skipping date: {date}")
            else:
                data[date] = [region_id, rgb_image[y1:y2, x1:x2], water_mask[y1:y2, x1:x2]]
                masks.append(water_mask[y1:y2, x1:x2])

        # Average all available masks
        data['avg'] = utils.avg_masks(masks)
        logger.debug(f"First directory: {dirs[0]}")

        # Create a dictionary to track monthly data
        month_dict = {}
        for key, value in data.items():
            if key != 'avg' and key[4:6] not in month_dict:
                month_dict[int(key[4:6])] = key
        logger.debug(f"Month dict: {str(month_dict)}")

        # Compile yearly data
        yearly = []
        none_array = np.full((y2 - y1, x2 - x1), None, dtype=object)
        for month in range(1, 13):
            if month in month_dict:
                yearly.append(data[month_dict[month]][2])
                logger.debug(f"Available month: {month_dict[month]}")
            else:
                yearly.append(none_array)

        # Stack the yearly data
        stacked_array = np.stack(yearly, axis=2)
        data['yearly'] = stacked_array

        # Save the data to a file
        parent_dir = f"{input_data.data_dir}/{sub_dir}"
        os.makedirs(parent_dir, exist_ok=True)
        file_path = f"{input_data.data_dir}/{sub_dir}/landsat_avg_water_index_{region_id}_{year}.pkl"
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
