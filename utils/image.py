import numpy as np

def bands_to_rgb(img, b_r, b_g, b_b, alpha=1.0, figsize=(10, 10)):
    """
    Convert an image to RGB using specified bands.

    Parameters:
    img (np.ndarray): Input image with multiple bands.
    b_r (int): Index for the red band.
    b_g (int): Index for the green band.
    b_b (int): Index for the blue band.
    alpha (float, optional): Scaling factor for intensity. Default is 1.0.
    figsize (tuple, optional): Size of the figure for plotting. Default is (10, 10).

    Returns:
    np.ndarray: RGB image.
    """
    # Extract and stack bands
    rgb = np.stack([img[b_r], img[b_g], img[b_b]], axis=-1)

    # Normalize and scale
    max_value = rgb.max()
    if (max_value > 0):
        rgb = (rgb / max_value) * alpha

    return rgb


def calculate_black_pixel_percentage(rgb):
    """
    Calculate the percentage of completely black pixels in an RGB image array.

    Parameters:
    - rgb: A numpy array representing an RGB image, where the image has been normalized
           and adjusted by a certain `alpha` value as per the operations provided.

    Returns:
    - percentage_black: The percentage of pixels in the image that are completely black.
    """
    # Identify black pixels (all RGB values are 0)
    is_black = np.all(rgb == 0, axis=-1)

    # Calculate the percentage of black pixels
    total_pixels = rgb.shape[0] * rgb.shape[1]  # Total number of pixels in the image
    black_pixels = np.sum(is_black)  # Total number of black pixels
    percentage_black = (black_pixels / total_pixels) * 100

    return percentage_black


def avg_masks(masks):
    """
    Calculate the average of multiple masks.

    Parameters:
    masks : list of np.ndarray
        List of masks to be averaged. Each mask should be a numpy array of the same shape.

    Returns:
    np.ndarray
        Array with the averaged mask values, rounded to two decimal places.
    """
    # Convert the list of masks to a numpy array
    masks_array = np.array(masks)

    # Sum the masks along the first axis
    sum_masks = np.sum(masks_array, axis=0)

    # Calculate the average mask
    average_mask = sum_masks / len(masks)

    # Round the averaged mask to two decimal places
    rounded_average_mask = np.round(average_mask, 2)

    return rounded_average_mask