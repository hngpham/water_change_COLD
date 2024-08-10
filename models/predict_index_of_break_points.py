import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


from xgboost import XGBClassifier

from waterchange.utils import get_logger
from waterchange.utils import read_pkl, create_file_path
from ._utils import data_distribution

logger = get_logger("DEBUG")


def _add_class(dfa, added_col='class', compare_col='incr', threshold=0.1):
    dfa[added_col] = 0
    dfa.loc[(dfa[compare_col] < threshold) & (dfa[compare_col] > -threshold), added_col] = 0 # stay the same
    dfa.loc[dfa[compare_col] > threshold, added_col] = 2 # increasing
    dfa.loc[dfa[compare_col] < -threshold, added_col] = 1 # Decreasing
    return dfa


def _add_class_(dfa, threshold=0):
    dfa['class'] = 0
    dfa.loc[(dfa['incr'] == threshold), 'class'] = 0 # stay the same
    dfa.loc[dfa['incr'] > threshold, 'class'] = 2 # increasing
    dfa.loc[dfa['incr'] < -threshold, 'class'] = 1 # Decreasing
    return dfa


def _extract_XY_with_scaler(df, max_index, scaler = None):
    X = df.iloc[:,6: (6+max_index)].to_numpy()
    if not scaler:
        scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    Y = df["class"].to_numpy()
    return X_normalized, Y, scaler

def _extract_XY_indices(df, start, end):
    X = df.iloc[:, start:end].to_numpy()
    Y = df["class"].to_numpy()
    return X, Y

def _extract_XY(df, max_index):
    """
    Decide to skip the scaler due to performance as shown here.
    With scaler           Without scaler
    Overall: 0.83 ± 0.01  Overall: 0.83 ± 0.01
    Class 0: 0.91 ± 0.03  Class 0: 0.91 ± 0.03
    Class 1: 0.65 ± 0.04  Class 1: 0.65 ± 0.04
    Class 2: 0.67 ± 0.11  Class 2: 0.67 ± 0.11
    Overall: 0.69 ± 0.12  Overall: 0.69 ± 0.16
    Class 0: 0.96 ± 0.02  Class 0: 0.86 ± 0.15
    Class 1: 0.20 ± 0.13  Class 1: 0.35 ± 0.26
    Class 2: 0.11 ± 0.12  Class 2: 0.32 ± 0.19
    """
    X = df.iloc[:,6:(6+max_index)].to_numpy()
    Y = df["class"].to_numpy()
    return X, Y, None


def _predict(model, X_test, y_test):
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    overall_accuracy = accuracy_score(y_test, predictions)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    # Calculate accuracy for each class
    class_accuracies = []
    for i in range(conf_matrix.shape[0]):
        class_accuracy = conf_matrix[i, i] / sum(conf_matrix[i, :])
        class_accuracies.append(class_accuracy)

    # Print accuracy for each class
    for i, accuracy in enumerate(class_accuracies):
        logger.debug(f"{i} Accuracy: {np.round(accuracy,3)}")

    # Optionally, you can print or visualize the confusion matrix

    logger.debug("Confusion Matrix:")
    logger.debug(conf_matrix)
    report = classification_report(y_test, y_pred)
    logger.debug(report)

    logger.info(f"Overall Accuracy: {overall_accuracy} - {str(class_accuracies)}")
    return predictions, [overall_accuracy] + class_accuracies


def _predict_regression(y_pred, y_test):

    overall_accuracy = accuracy_score(y_test, y_pred)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Calculate accuracy for each class
    class_accuracies = []
    for i in range(conf_matrix.shape[0]):
        class_accuracy = conf_matrix[i, i] / sum(conf_matrix[i, :])
        class_accuracies.append(class_accuracy)

    # Print accuracy for each class
    for i, accuracy in enumerate(class_accuracies):
        logger.debug(f"{i} Accuracy: {np.round(accuracy,3)}")

    # Optionally, you can print or visualize the confusion matrix

    logger.debug("Confusion Matrix:")
    logger.debug(conf_matrix)
    report = classification_report(y_test, y_pred)
    logger.debug(report)

    logger.info(f"Overall Accuracy: {overall_accuracy} - {str(class_accuracies)}")
    return y_pred, [overall_accuracy] + class_accuracies


def predict_average_water_change_at_break_point(
    input_data,
    sub_dir_in_train: str,
    train_region_id: List[int],
    infer_region_id: int,
    threshold: float = 0.1) -> Tuple[List[int], List[float]]:
    df_train = pd.DataFrame()  # Initialize an empty DataFrame instead of a list
    for index in train_region_id:
        file_path = create_file_path(input_data.data_dir, sub_dir_in_train, f"water_coefs_break_point_df_region_{index}.pkl")
        read_df = read_pkl(file_path)
        df_train = pd.concat([df_train, read_df], axis=0)  # Concatenate DataFrames

    file_path = create_file_path(input_data.data_dir, sub_dir_in_train, f"water_coefs_break_point_df_region_{infer_region_id}.pkl")
    df_infer = read_pkl(file_path)

    max_index = 112
    df_train = _add_class(df_train, threshold=0.1)
    X, Y, _ = _extract_XY(df_train, max_index)

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model no training data
    model = XGBClassifier()
    model.fit(X_train, y_train, verbose=False)

    logger.info("Training:")
    predict_train, acc_train = _predict(model, X_test, y_test)

    df_infer = _add_class(df_infer, 'class', 'incr', threshold)
    dist_infer = data_distribution(df_infer)
    X_infer, Y_infer, _ = _extract_XY(df_infer, max_index)

    logger.info("Inference")
    predicts_infer, acc_infer = _predict(model, X_infer, Y_infer)

    return predicts_infer, acc_train, acc_infer, dist_infer


def predict_average_water_change_at_break_point_from_trained_model(
    input_data,
    model,
    sub_dir_in_train: str,
    infer_region_id: int,
    threshold: float = 0.1) -> Tuple[List[int], List[float], Dict[int, int]]:

    file_path = create_file_path(input_data.data_dir, sub_dir_in_train, f"water_coefs_break_point_df_region_{infer_region_id}.pkl")
    df_infer = read_pkl(file_path)

    df_infer = _add_class(df_infer, 'class', 'incr', threshold)
    dist_infer = data_distribution(df_infer)

    X_infer_prev, Y_infer = _extract_XY_indices(df_infer, 6, 62)
    X_infer_cur , _       = _extract_XY_indices(df_infer, 62, 118)

    df_infer["pred_prev"] = model.predict(X_infer_prev)
    df_infer["pred_cur"]  = model.predict(X_infer_cur)
    df_infer["pred_incr"] = df_infer["pred_cur"] - df_infer["pred_prev"]

    df_infer = _add_class(df_infer, 'class_pred', 'pred_incr', threshold)

    logger.info("Inference")
    predicts_infer, acc_infer = _predict_regression(df_infer["class_pred"], df_infer["class"])

    return predicts_infer, acc_infer, dist_infer