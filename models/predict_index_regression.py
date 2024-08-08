import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from waterchange.utils import read_pkl, create_file_path
from waterchange.utils import get_logger
from ._utils import data_distribution

logger = get_logger("DEBUG")

def _add_class(df):
    # intervals = [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)]
    intervals = [(0.0, 0.3), (0.3, 1.0)]
    conditions = []
    counter = int(0)
    table = {}
    total = 0
    for i in intervals:
        condition = (df["avg"] >= i[0]) & (df["avg"] <= i[1])
        df.loc[condition, "class"] = int(counter)
        logger.debug(f"{counter}, {i}")
        table[counter] = [i]
        counter += 1
    return df, table


def get_bands_from_df(df, band_1_indexed: int = 3, band_2_indexed: int = 5):
    # Adjusting for 0-based indexing
    band_1_idx = band_1_indexed - 1
    band_2_idx = band_2_indexed - 1

    # Reshape the dataframe
    reshaped_df = df.reshape((-1, 8, 7))

    # Extract the bands
    band_1 = reshaped_df[:, :, band_1_idx]
    band_2 = reshaped_df[:, :, band_2_idx]

    # Stack and reshape
    combined = np.stack((band_1, band_2), axis=2).reshape(df.shape[0], -1)

    return combined


def _extract_XY_bands(df, band_1_1st_indexed: int = 3, band_2_1st_indexed: int = 5):
    # print(df.iloc[:,4:60].head(1))
    X = df.iloc[:,4:60].to_numpy()
    X_bands = get_bands_from_df(X, band_1_1st_indexed, band_2_1st_indexed)

    # [ 0.  1.  3.  4.  5.  7.  8. 14. 17. 23. 26.]
    logger.debug(set(df["class"]))
    logger.debug(f"Number of class: {len(set(df['class']))}")

    le = LabelEncoder()
    df["class"] = le.fit_transform(df["class"])
    le_name_mapping = dict(zip(le.classes_, le.inverse_transform(le.transform(le.classes_))))
    Y = df["class"].to_numpy()
    logger.debug("New labels to old labels:")
    logger.debug(le_name_mapping)
    return X_bands, Y, le_name_mapping


def _extract_XY_all_bands(df):
    X = df.iloc[:,4:60].to_numpy() # position, [start_year, end_year], water_list, calculate_average(water), coefs, 'position', 'time', 'water', 'avg'
    Y = df["avg"].to_numpy()

    return X, Y


def _predict(model, X_test, y_test: Optional[np.ndarray] = None) -> Tuple[List[float], float, float]:
    # make predictions for test data
    y_pred = model.predict(X_test)
    if y_test is not None:
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return y_pred, mae, r2
    else:
        return y_pred


def predict_average_water_index_all_pixels_from_first_coefs_all_bands_regression(
    model_name: str,
    input_data,
    sub_dir_in_train: str,
    sub_dir_in_infer_first_all: str,
    sub_dir_in_infer_in_period: str,
    train_region_id: List[int],
    infer_region_id: int) -> Tuple[pd.DataFrame, List[int], List[int]]:

    df_train = pd.DataFrame()  # Initialize an empty DataFrame instead of a list
    for index in train_region_id:
        file_path = create_file_path(input_data.data_dir, sub_dir_in_train, f"water_coefs_df_region_{index}.pkl")
        read_df = read_pkl(file_path)
        df_train = pd.concat([df_train, read_df], axis=0)  # Concatenate DataFrames

    file_path = create_file_path(input_data.data_dir, sub_dir_in_infer_first_all, f"first_coefs_all_pixels_region_{infer_region_id}.pkl")
    df_infer_all_pixels = read_pkl(file_path)

    file_path = create_file_path(input_data.data_dir, sub_dir_in_infer_in_period, f"water_coefs_df_region_{infer_region_id}.pkl")
    df_infer_pixels_in_period = read_pkl(file_path)

    # df_train, table = _add_class(df_train)
    # data_distribution(df_train, table)
    X, Y = _extract_XY_all_bands(df_train)

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # Inference on pixels in period
    X_infer_in_period, Y_infer_in_period = _extract_XY_all_bands(df_infer_pixels_in_period)
    # Inference on all pixels in different period
    X_infer_all_pixels = df_infer_all_pixels.iloc[:,2:58].to_numpy()

    models = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(),
        "random_forest": RandomForestRegressor(),
        "gradient_boosting": GradientBoostingRegressor()
    }

    # Iterate over models and evaluate

    model = models[model_name]
    model.fit(X_train, y_train)
    # Train and evaluate on training data
    _, mae_train, r2_train = _predict(model, X_test, y_test)
    logger.debug(f"{model_name} - Training error: MAE = {mae_train}, R2 = {r2_train}")

    # Evaluate on inference data
    predict_infer_pixels_in_period, mae_infer_pixels_in_period, r2_infer_pixels_in_period = _predict(model, X_infer_in_period, Y_infer_in_period)
    logger.debug(f"{model_name} - Inference error: MAE = {mae_infer_pixels_in_period}, R2 = {r2_infer_pixels_in_period}")

    # Make predictions on all inference pixels
    y_pred = _predict(model, X_infer_all_pixels)
    df_infer_all_pixels["pred_avg"] = y_pred

    # Set 'pred' to 1 where 'pred_avg' is greater than or equal to 0.3, otherwise set to 0
    df_infer_all_pixels["pred"] = (df_infer_all_pixels["pred_avg"] >= 0.3).astype(int)


    for index, row in df_infer_all_pixels.iterrows():
        if row['time'][0] < 1984 or row['time'][1] > 2000:
            df_infer_all_pixels.at[index, 'pred'] += 3

    return df_infer_all_pixels
    # return train_result, infer_result, mse_train, mse_infer_pixels_in_period # class_accuracies + [overall_accuracy]


def predict_average_water_index_all_pixels_from_first_coefs_bands(
    input_data,
    sub_dir_in_train: str,
    sub_dir_in_infer_first_all: str,
    sub_dir_in_infer_in_period: str,
    train_region_id: List[int],
    infer_region_id: int,
    band1: int = 3,
    band2: int = 5) -> Tuple[pd.DataFrame, List[int], List[int]]:

    df_train = pd.DataFrame()  # Initialize an empty DataFrame instead of a list
    for index in train_region_id:
        file_path = create_file_path(input_data.data_dir, sub_dir_in_train, f"water_coefs_df_region_{index}.pkl")
        read_df = read_pkl(file_path)
        df_train = pd.concat([df_train, read_df], axis=0)  # Concatenate DataFrames

    file_path = create_file_path(input_data.data_dir, sub_dir_in_infer_first_all, f"first_coefs_all_pixels_region_{infer_region_id}.pkl")
    df_infer_all_pixels = read_pkl(file_path)

    file_path = create_file_path(input_data.data_dir, sub_dir_in_infer_in_period, f"water_coefs_df_region_{infer_region_id}.pkl")
    df_infer_pixels_in_period = read_pkl(file_path)

    df_train, table = _add_class(df_train)
    data_distribution(df_train, table)
    X, Y, le_name_mapping = _extract_XY_bands(df_train, band1, band2)

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model no training data
    model = XGBClassifier()
    model.fit(X_train, y_train, verbose=False)

    # Inference on all pixels
    logger.info("Training:")
    predict_train, train_acc = _predict(model, X_test, y_test, le_name_mapping, table)

    # Inference on pixels in period
    logger.info("Inference:")
    df_infer_pixels_in_period, table = _add_class(df_infer_pixels_in_period)
    data_distribution(df_infer_pixels_in_period, table)
    X_infer_in_period, Y_infer_in_period, le_name_mapping = _extract_XY_bands(df_infer_pixels_in_period, band1, band2)
    predict_infer_pixels_in_period, infer_pixels_in_period_acc = _predict(model, X_infer_in_period, Y_infer_in_period, le_name_mapping, table)

    X_infer_all_pixels = df_infer_all_pixels.iloc[:,2:58].to_numpy()
    X_infer_all_pixels = get_bands_from_df(X_infer_all_pixels, band1, band2)
    y_pred = model.predict(X_infer_all_pixels)
    df_infer_all_pixels["pred"] = y_pred

    for index, row in df_infer_all_pixels.iterrows():
        if row['time'][0] < 1984 or row['time'][1] > 2000:
            df_infer_all_pixels.at[index, 'pred'] += 3

    return df_infer_all_pixels, train_acc, infer_pixels_in_period_acc # class_accuracies + [overall_accuracy]
