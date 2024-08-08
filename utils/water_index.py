import numpy as np

def normalized_difference(img, b1, b2, eps=0.0001):
    """
    Calculate the normalized difference index for two bands.
    Landsat 5
    ndvi  -> (img, 2, 4)
    mndwi -> (img, 2, 5)

    Parameters:
    img : dict
        A dictionary containing image bands as numpy arrays.
    b1 : str
        Key for the first band in the img dictionary.
    b2 : str
        Key for the second band in the img dictionary.
    eps : float, optional
        A small value to avoid division by zero, by default 0.0001.

    Returns:
    np.ndarray
        Array with the normalized difference index values.
    """
    band1 = np.where((img[b1]==0) & (img[b2]==0), np.nan, img[b1])
    band2 = np.where((img[b1]==0) & (img[b2]==0), np.nan, img[b2])
    
    return (band1 - band2) / (band1 + band2 + eps)