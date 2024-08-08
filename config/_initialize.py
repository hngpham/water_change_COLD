from omegaconf import OmegaConf
import os

from waterchange.utils.logging import get_logger

logger = get_logger("DEBUG")

def read_config():
    # Assuming config.yaml is in the root of the waterchange package
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yaml')
    logger.debug(f"Loading {config_path}.")

    # Check if the config_path exists
    if os.path.exists(config_path):
        # Load the config.yaml file
        config = OmegaConf.load(config_path)
        logger.debug("Config loaded successfully.")
    else:
        logger.error(f"Config file {config_path} does not exist.")
    return config

def test():
    logger.debug("Automatic reload!")