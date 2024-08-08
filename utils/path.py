from omegaconf import OmegaConf
import glob
import os

from datetime import datetime
from datetime import timedelta

from waterchange import utils

logger = utils.get_logger("DEBUG")

def get_sorted_paths_list(config: OmegaConf) -> list:
    """
    Get all directories to pixel data files (.mat).
    The file names are chosen with 'pattern' and
    sorted by the part of their names specified by the 'sort_key'.

    For example:
        - TSLine
        - TSLine1

    Parameters
    ----------
    config : OmegaConf
        Config file can be loaded in from the config.yaml
    Returns
    -------
    list
        Absolute directories.
    """
    logger.debug("Getting sorted paths to pixel data points.")
    file_dirs = []

    for folder in config.files.values():
        logger.debug(f"Processing folder: {folder.path}\nFolder pattern: {folder.pattern}\nSort key: {folder.sort_key}")
        found_dirs = glob.glob(f"{folder.path}/{folder.pattern}")

        logger.debug(f"Found {len(found_dirs)} files")
        sorted_found_dirs = sorted(
            found_dirs,
            key = lambda x: x[folder.sort_key[0]:folder.sort_key[1]]
        )
        logger.debug(f"First and last directory in this folder \
                     \n {sorted_found_dirs[0]} \
                     \n {sorted_found_dirs[-1]}")

        file_dirs += sorted_found_dirs

    logger.debug(f"Found {len(file_dirs)} in COMBINED folder.")
    logger.debug(f"First and last directory in COMBINED folders \
                \n {file_dirs[0]} \
                \n {file_dirs[-1]}")

    return file_dirs


def get_all_subdirectories(directory: str) -> list:
    """
    Get all subdirectories within a specified directory.

    Parameters
    ----------
    directory : str
        The path to the directory in which to search for subdirectories.

    Returns
    -------
    list
        A sorted list of paths to subdirectories within the specified directory.
    """
    subdirectories = [os.path.join(directory, d)
                      for d in os.listdir(directory)
                      if os.path.isdir(os.path.join(directory, d))]
    return sorted(subdirectories)


def create_file_path(
    parent_dir: str,
    sub_dir: str,
    file_name: str) -> str:
    """
    Constructs a file path by joining a base directory with a subdirectory and filename.
    Creates the subdirectory if it doesn't exist.

    Parameters
    ----------
    input_data : Any
        An object that contains a 'data_dir' attribute representing the base directory.
    sub_dir : str
        The name of the subdirectory where the file will be stored.
    file_name : str
        The name of the file to be created or accessed.

    Returns
    -------
    str
        The full path to the file, including the base directory, subdirectory, and filename.
    """
    # Construct the subdirectory path
    sub_dir_path = os.path.join(parent_dir, sub_dir)

    # Ensure the subdirectory exists
    os.makedirs(sub_dir_path, exist_ok=True)

    # Create the full file path
    file_path = os.path.join(sub_dir_path, file_name)

    return file_path