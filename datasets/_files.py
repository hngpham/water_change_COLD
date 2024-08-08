from omegaconf import OmegaConf
import os

from waterchange.utils import get_logger, get_sorted_paths_list

logger = get_logger("DEBUG")

class DataInput:
    def __init__(self, paths, regions, rois, data_dir, landsat_dir, viz_dir):
        self.paths = paths
        self.regions = regions
        self.rois = rois
        self.data_dir = data_dir
        self.landsat_dir = landsat_dir
        self.viz_dir = viz_dir

def setup_directory(config):
    # Each file contains data for a row including 5000 pixels in TSLine folder.
    # File names numbered from 1 to 5000 ---> index 0 -> 4999
    paths = get_sorted_paths_list(config)

    regions = list(config.regions)
    logger.info(f"Found {len(regions)} regions.")

    rois = config.rois
    logger.info(f"Indices of regions of interest: {str(rois)}")

    landsat_dir = config.landsat_dir
    logger.info(f"Landsat data at {landsat_dir}")

    out_dir = config.output_dir
    data_dir = out_dir + "/data"
    viz_dir = out_dir + "/viz"

    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"Output folder for data files: {data_dir}")

    os.makedirs(viz_dir, exist_ok=True)
    logger.info(f"Output folder for visualization files: {viz_dir}")

    data_input = DataInput(paths, regions, rois, data_dir, landsat_dir, viz_dir)
    return data_input

