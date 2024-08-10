import pandas as pd
import numpy as np
import scipy.io

from tqdm import tqdm
from typing import List, Tuple, Dict

from ._files import DataInput
from ._utils import datenum_to_datetime, calculate_average
from waterchange import utils

logger = utils.get_logger("DEBUG")

class RowTsline:
    """
    Extract from the .mat file in TSLine folder.
    Each file is data of a pixel. 
    """
    def __init__(self, path: str):
        self.path = path
        self.data = scipy.io.loadmat(path)["rec_cg"][0]

    def __getitem__(self, key: int):
        t_start = self.data["t_start"][key].tolist()[0][0]
        t_end = self.data["t_end"][key].tolist()[0][0]
        t_break = self.data["t_break"][key].tolist()[0][0]
        time = (t_start, t_end, t_break)
        pos = self.data["pos"][key].tolist()[0][0]
        coefs = self.data["coefs"][key]
        num_obs = self.data["num_obs"][key].tolist()[0][0]
        return pos, coefs, num_obs, time

def extract_first_coeffcients_from_mat_files(
    input_data: DataInput,
    start_row: int,
    end_row: int, count: int = 1
) -> List[Tuple]:
    """
    Extract the FIRST coefficients of each pixel
    """
    first_coefs = []
    limit = len(input_data.paths)
    logger.debug(f"Extracting FIRST coefficients of each pixel in the COLD data.")
    for r in tqdm(range(start_row, end_row)):
        row = []
        # get the first pos data
        rowData = RowTsline(input_data.paths[r])
        for pos, coefs, num_obs, time in rowData:
            if pos % limit > limit:
                count = limit*(r+1) + 1
                break
            if pos == count:
                row.append((pos, coefs, num_obs, time))
                # print(r, pos, time)
                count += 1
                # print(r, pos)
        first_coefs.append(tuple(row))
    
    logger.debug(f"Done.")
    return first_coefs


def extract_all_coeffcients_from_mat_files(
    input_data, start_row: int,
    end_row: int,
    count: int = 1
) -> List[Tuple]:
    """
    Extract ALL coefficients of each pixel
    start_row, end_row are inclusive
    Each row includes:
        pos_list, coefs_list, numb_obs_list, time_list

    """
    # file names numbered from 1 to 5000 ---> index 0 -> 4999
    all_coefs = []
    limit = len(input_data.paths) # 5000 in TSLine data
    logger.debug(f"Extracting ALL coefficients of each pixel in the COLD data.")
    for r in tqdm(range(start_row, end_row)):
        rowData = RowTsline(input_data.paths[r])
        row = []
        coefs_list = []
        numb_obs_list = []
        time_list = []
        pos_list = []
        for pos, coefs, num_obs, time in rowData:
            # print("index pos - count", pos, count)
            if pos == count:
                # print("push at position ",count)
                coefs_list.append(coefs)
                numb_obs_list.append(num_obs)
                time_list.append(time)
                pos_list.append(pos)
            else:
                row.append((pos_list, coefs_list, numb_obs_list, time_list))
                # print(r, pos_list, time_list)
                # print("push at row", r, pos_list)
                count += 1
                # print("increase count to ", count, "and push at", count)
                coefs_list = [coefs]
                numb_obs_list = [num_obs]
                time_list = [time]
                pos_list = [pos]

        if pos%limit == 0:
            # print("push 5000")
            row.append((pos_list, coefs_list, numb_obs_list, time_list))
            # print(r, pos_list, time_list)
            count += 1
            # print("increase count to ", count, "and push at", count)

        all_coefs.append(tuple(row))

    logger.debug(f"Done.")
    return all_coefs


def extract_sub_region(
    coefs: List[Tuple],
    coordinates: List
) -> List[Tuple]:
    """
    Crop out smaller regions from the parent.
    Parameters
    ----------
    coefs : List[Tuple]
        Extracted coefficients 'all_coefs' or 'first_coefs'
    coordinates : List
      (y1, x1)
        +-----------+ (y1, x2)
        |           |
        |           |
        +-----------+ (y2, x2)
        (y2, x1)

        y-axis: vertical
        x-axis: horizontal

    Returns
    -------
    List[Tuple]
        COLD coefficients
    """
    y1, y2, x1, x2 = coordinates
    logger.debug(f"Sub-region's coordinates: {str(coordinates)}")
    sub_region = [row[x1:x2] for row in coefs[y1:y2]]
    return sub_region


def extract_first_coefs_all_pixels_to_df(
    input_data,
    sub_dir_in: str = "all_coefs",
    sub_dir_out : str = "first_coefs_all_pixels",
    region_ids: List[int] = [0]
) -> None:
    
    for region_id in region_ids:
        logger.debug(f"Extract coefs within a period in region {region_id}.")
        file_path = utils.create_file_path(input_data.data_dir, sub_dir_in, f"all_coefs_region_{region_id}.pkl")
        logger.info(f"Loaded file at {file_path}")
        sub_region = utils.read_pkl(file_path)
        logger.debug(f"Dim: {len(sub_region)} x {len(sub_region[0])}")

        count = 0
        allYears = []
        r = 0
        t = 0
        for row in sub_region:
            # logger.debug(f"row {r}")
            for tuple_ in row:
                # logger.debug(f"tuple {t}")
                # Select the first coefs
                i = 0
                # logger.debug(tuple_[0][i])
                start_year = datenum_to_datetime(tuple_[3][i][0]).year
                end_year   = datenum_to_datetime(tuple_[3][i][1]).year
                # logger.debug(tuple_[0][i], start_year, end_year)
                coefs =  np.array(tuple_[1][i]).reshape(56)
                temp = [tuple_[0][i], [start_year, end_year]]
                for coef in coefs:
                    temp.append(coef)
                allYears.append(temp)
                count += 1
                t+=1
            r+=1

        columns = ['position', 'time'] + [str(i) for i in range(56)]
        df = pd.DataFrame(allYears, columns=columns)

        file_path = utils.create_file_path(input_data.data_dir, sub_dir_out, f"first_coefs_all_pixels_region_{region_id}.pkl")
        df.to_pickle(file_path)
        logger.info(f"Saved file at {file_path}")


def extract_water_index_coefs_within_period_to_df(
    input_data, avg_water_subdir: str = "avg_water_index",
    all_coefs_subdir: str = "all_coefs",
    sub_dir_out: str = "water_coefs_in_period",
    region_id: int = 0, data_range = [1984, 2000]
) -> None:
    """
    24001370, 1997 1999, [None, None, None, None, None, None, None, None, None, None, None, None, None, 0.0, 0.0, 0.0, None] average, coefs
    """
    cold = {}
    y1, y2, x1, x2 = input_data.regions[region_id]
    
    for year in range(data_range[0], data_range[1] + 1):  # Loop from 1985 to 2000
        file_path = f"{input_data.data_dir}/{avg_water_subdir}/landsat_avg_water_index_{region_id}_{year}.pkl"
        cold[year] = np.round(utils.read_pkl(file_path)['avg'].reshape((y2 - y1)*(x2 - x1)), 2)

    cold_df = pd.DataFrame(cold)

    logger.debug("Extract coefs within a period.")
    file_path = f"{input_data.data_dir}/{all_coefs_subdir}/all_coefs_region_{region_id}.pkl"

    sub_region = utils.read_pkl(file_path)
    logger.debug(f"Dim: {len(sub_region)} x {len(sub_region[0])}")

    count = 0
    allYears = []
    r = 0
    t = 0
    for row in sub_region:
        # print("row ", r)
        for tuple_ in row:
            # print("tuple ", t)
            for i in range(len(tuple_[0])):
                # print(tuple_[0][i])
                start_year = datenum_to_datetime(tuple_[3][i][0]).year
                end_year   = datenum_to_datetime(tuple_[3][i][1]).year
                if start_year >= data_range[0] and end_year <= data_range[1]:
                    # print(tuple_[0][i], start_year, end_year)
                    numsYears = 0
                    water = [None] * (data_range[1]-data_range[0]+1)
                    for year in range(start_year, end_year+1):
                        water[year - data_range[0]] = cold_df[year][count]
                    print(tuple_[0][i], start_year, end_year, water)
                    coefs =  np.array(tuple_[1][i]).reshape(56)
                    temp = [tuple_[0][i], [start_year, end_year], water, calculate_average(water)]
                    for coef in coefs:
                        temp.append(coef)
                    allYears.append(temp)
            count += 1
            # t+=1
        # r+=1

    columns = ['position', 'time', 'water', 'avg'] + [str(i) for i in range(56)]
    df = pd.DataFrame(allYears, columns=columns)

    file_path = utils.create_file_path(input_data.data_dir, sub_dir_out, f"water_coefs_df_region_{region_id}.pkl")
    df.to_pickle(file_path)
    logger.info(f"Saved file at {file_path}")


def extract_break_point_from_water_coefs_df(
    input_data: DataInput,
    sub_dir_in: str,
    sub_dir_out: str,
    region_id: int) -> None:
    """
                  0,    1,    2,        3,            4,            5,        6-61,      62-117
    Output: 9501861, prev, curr, increase, [1988, 1989], [1989, 1991], prev[coefs], curr[coefs]
    """
    file_path_in = utils.create_file_path(input_data.data_dir, sub_dir_in, f"water_coefs_df_region_{region_id}.pkl")
    df = pd.read_pickle(file_path_in)

    duplicates = df[df.duplicated('position', keep=False)]
    for row in duplicates.iterrows():
        print(row[1]['position'], row[1]['time'], row[1]['avg'], row[1]['water'])
        # 9501861 [1988, 1989] 0.0 [None, None, None, None, 0.0, 0.0, None, None, None, None, None, None, None, None, None, None, None]
    data = []
    for idx in range(1, df.shape[0]):
        prev = df.iloc[idx-1]
        curr = df.iloc[idx]
        if prev['position'] == curr['position']:
            increase = curr['avg'] - prev['avg']
            print(curr['position'], prev['avg'], curr['avg'], increase, prev['time'], curr['time'])
            # 9501861 0.0 0.0 0.0 [1988, 1989] [1989, 1991]
            data.append([curr['position'], prev['avg'], curr['avg'], increase, prev['time'], curr['time']] + prev[4:60].tolist() + curr[4:60].tolist())
    columns = ['position', 'prev', 'cur', 'incr', 'prevT', 'curT'] + [x for x in range(112)]
    df_res = pd.DataFrame(data, columns=columns)

    file_path = utils.create_file_path(input_data.data_dir, sub_dir_out, f"water_coefs_break_point_df_region_{region_id}.pkl")
    df_res.to_pickle(file_path)

    logger.info(f"Saved file at {file_path}")

