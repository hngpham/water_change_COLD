import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from waterchange.utils import get_logger
from waterchange.utils import read_pkl, create_file_path
from ._utils import data_distribution

logger = get_logger("DEBUG")


def _add_class(dfa, threshold):
    dfa['class'] = 0
    dfa.loc[(dfa['incr'] < threshold) & (dfa['incr'] > -threshold), 'class'] = 0 # stay the same
    dfa.loc[dfa['incr'] > threshold, 'class'] = 2 # increasing
    dfa.loc[dfa['incr'] < -threshold, 'class'] = 1 # Decreasing
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


def _extract_XY(df, max_index):
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


def predict_average_water_change_at_break_point(
    input_data,
    sub_dir_in_train: str,
    train_region_id: List[int],
    infer_region_id: int) -> Tuple[List[int], List[float]]:
    df_train = pd.DataFrame()  # Initialize an empty DataFrame instead of a list
    for index in train_region_id:
        file_path = create_file_path(input_data.data_dir, sub_dir_in_train, f"water_coefs_break_point_df_region_{index}.pkl")
        read_df = read_pkl(file_path)
        df_train = pd.concat([df_train, read_df], axis=0)  # Concatenate DataFrames

    file_path = create_file_path(input_data.data_dir, sub_dir_in_train, f"water_coefs_break_point_df_region_{infer_region_id}.pkl")
    df_infer = read_pkl(file_path)

    threshold = 0.1
    max_index = 112
    df_train = _add_class(df_train, threshold=0.1)
    X, Y, scaler = _extract_XY(df_train, max_index)

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model no training data
    model = XGBClassifier()
    model.fit(X_train, y_train, verbose=False)

    logger.info("Training:")
    predict_train, acc_train = _predict(model, X_test, y_test)

    df_infer = _add_class(df_infer, threshold)
    # data_distribution(df_infer)
    X_infer, Y_infer, _ = _extract_XY(df_infer, max_index)

    logger.info("Inference")
    predicts_infer, acc_infer = _predict(model, X_infer, Y_infer)

    return predicts_infer, acc_train, acc_infer