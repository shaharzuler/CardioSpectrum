import numpy as np

def nansem(arr):
    valid_values = arr[~np.isnan(arr)]
    return np.nanstd(valid_values, ddof=1) / np.sqrt(len(valid_values))

def nanstd(arr):
    valid_values = arr[~np.isnan(arr)]
    return np.nanstd(valid_values)
