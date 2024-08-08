import glob
import os
import numpy as np

from omegaconf import OmegaConf

from datetime import datetime
from datetime import timedelta

from waterchange.utils import get_logger

logger = get_logger("DEBUG")


def calculate_average(lst):
    valid_values = [x for x in lst if x is not None]
    if not valid_values:
        return None
    return np.mean(valid_values)


def datenum_to_datetime(datenum: int) -> datetime:
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    return datetime.fromordinal(int(datenum)) \
           + timedelta(days=days) \
           - timedelta(days=366)


