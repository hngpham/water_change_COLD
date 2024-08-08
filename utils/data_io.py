import os
import pickle
from tqdm import tqdm
from typing import List, Tuple

from waterchange.utils.path import create_file_path
from waterchange.utils.logging import get_logger

logger = get_logger("DEBUG")


def save_pkl(
    input_data: str,
    sub_dir: str, data: List[Tuple],
    name_base: str,
    number_of_splits: int = 1
) -> None:
    """
    Save coefficients file into multiple splits to 'output_dir/data'
    Parameters
    ----------
    data : List[Tuple]
        all_coefs or first_coefs of the extracted COLD data.
    name_base : str
    number_of_splits : int, optional
        In case, file is too large, by default 1
    """
    # Calculate the size of each split
    split_size = len(data) // number_of_splits
    logger.debug(f"Exporting {name_base} with {number_of_splits} splits.")

    for i in tqdm(range(number_of_splits), desc="Saving splits"):
        # Determine the start and end indices for the split
        start_idx = i * split_size
        # Handle the last split, which might be larger due to rounding
        if i == number_of_splits - 1:
            end_idx = len(data)
        else:
            end_idx = (i + 1) * split_size

        # Get the data subset
        data_split = data[start_idx:end_idx]

        if number_of_splits == 1:
            file_name = f"{name_base}.pkl"
        else:
            file_name = f"{name_base}_part_{i + 1}.pkl"

        file_path = create_file_path(input_data.data_dir, sub_dir, file_name)

        # Save the data split to a pickle file
        with open(file_path, 'wb') as file:
            pickle.dump(data_split, file)

        logger.info(f"Saved file at {file_path}")

    logger.debug(f"Done.")


def read_pkl(path: str) -> List[Tuple]:
    """
    Read pickle file of coefficients
    Parameters
    ----------
    path : str
        _description_

    Returns
    -------
    List[Tuple]
        COLD coefficients
    """
    with open(path, 'rb') as file:
        sub_region = pickle.load(file)
    return sub_region