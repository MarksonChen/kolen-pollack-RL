import numpy as np

# Sample 2D array
arr = np.array([
    [1, 2, 3],
    [np.nan, np.nan, np.nan],
    [4, 5, 6],
    [7, 8, np.nan]
])

# Function to compute standard deviation with np.nan for rows filled with np.nan
def row_std_with_nan(arr):
    # Identify rows filled with np.nan
    # nan_rows = np.all(np.isnan(arr), axis=1)
    # Compute the standard deviation for each row
    std_devs = np.nanstd(arr, axis=1)
    # Replace standard deviation of rows filled with np.nan with np.nan
    # std_devs[nan_rows] = np.nan
    return std_devs

# Compute the result
result = row_std_with_nan(arr)
print(result)
