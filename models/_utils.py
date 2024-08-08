import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from typing import Dict

import scipy.stats as stats

from waterchange.utils import create_file_path
from waterchange.utils import get_logger


logger = get_logger("DEBUG")

def get_base(n):
    q = n // 5000
    base = q * 5000
    return base


def data_distribution(df, table):
    hist, bins = np.histogram(df["class"], bins='auto')

    # Plot the histogram
    plt.hist(df["class"], bins='auto', alpha=0.7, color='blue', edgecolor='black')

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Values')

    logger.debug(df["class"].value_counts().sort_index())

    class_counts = df["class"].value_counts().sort_index() # Get counts of each class
    total_samples = len(df)  # Total number of samples in the DataFrame
    # Calculate percentage for each class
    class_percentages = (class_counts / total_samples) * 100
    # Create a dictionary with class as keys and percentages as values
    result_dict = class_percentages.to_dict()
    for key, value in result_dict.items():
        logger.debug("Table key: %s, Value: %s", table[key], round(value, 2))


def _get_row_number_and_start(number, first_pos, base):
    """
    Mapping the first position to coordinate (r = 0, c = 0)
    """
    offset = number - base
    # 2151 = 24002151 - 24000000

    rowNum = offset // 5000
    # 0 = 2151 // 5000

    start  = first_pos + rowNum * 5000
    # 24002151 = 24002151 + 0

    colNum = number - start
    # 0 = 24002151 - 24002151

    return rowNum, colNum

def create_inference_map(
    input_data,
    sub_dir_out: str,
    region_id: int,
    inference_df: pd.DataFrame,
    encoded_colors = Dict[int, str],
    empty_index: int = 3,
    ) -> None:
    # Initialize inference_map with value 3
    height = input_data.regions[region_id][1] - input_data.regions[region_id][0]
    width = input_data.regions[region_id][3] - input_data.regions[region_id][2]

    combined_map = np.full((height, width), empty_index)

    curPos = 0

    first_pos = inference_df.iloc[0]['position']
    base = get_base(first_pos)

    for index, row in inference_df.iterrows():
        if row["position"] != curPos:
            rowNum, colNum  =  _get_row_number_and_start(row["position"], first_pos, base)
            combined_map[rowNum][colNum] = row["pred"]
            curPos = row["position"]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 15))

    # Plot the array
    for i in range(combined_map.shape[0]):
        for j in range(combined_map.shape[1]):
            ax.add_patch(plt.Rectangle((j, -i-1), 1, 1, color=encoded_colors[combined_map[i, j]]))

    # Set limits and aspect ratio
    ax.set_xlim(0, combined_map.shape[1])
    ax.set_ylim(-combined_map.shape[0], 0)
    ax.set_aspect('equal')

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    file_path = create_file_path(input_data.viz_dir, sub_dir_out, f"inferred_map_region_{region_id}.png")
    plt.savefig(file_path, bbox_inches='tight')
    # Show plot
    plt.show()


def acc_stat(res):
    res = np.array(res)
    # Column names
    columns = ["Overall", "Class 0", "Class 1", "Class 2"]

    # Calculate the mean along each column
    mean_columns = np.mean(res, axis=0)

    # Calculate the standard deviation along each column
    std_columns = np.std(res, axis=0, ddof=1)  # using ddof=1 for sample standard deviation

    # Calculate the standard error of the mean
    sem_columns = std_columns / np.sqrt(res.shape[0])

    # Calculate the 95% confidence interval
    confidence_interval = stats.t.interval(0.95, df=res.shape[0]-1, loc=mean_columns, scale=sem_columns)

    # Format the results as mean ± 95% CI
    for i in range(len(mean_columns)):
        mean = mean_columns[i]
        ci_lower = confidence_interval[0][i]
        ci_upper = confidence_interval[1][i]
        ci = ci_upper - mean
        print(f"{columns[i]}: {mean:.2f} ± {ci:.2f}")

    return mean_columns, confidence_interval


def cal_p_value(res1, res2):
    # Ensure inputs are numpy arrays
    res1 = np.array(res1)
    res2 = np.array(res2)

    # Perform a two-sample t-test
    t_stat, p_value = stats.ttest_ind(res1[:, 0], res2[:, 0])

    # Print the t-statistic and p-value
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

    # Determine if the means are significantly different
    if p_value < 0.05:
        print("The means are significantly different (p < 0.05).")
    else:
        print("The means are not significantly different (p >= 0.05).")