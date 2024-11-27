import numpy as np
from sklearn.decomposition import PCA
import numpy as np
from scipy.interpolate import interp1d


def mean_imputation(data, axis, mask):
    """
    Perform mean imputation on a dictionary of 2D arrays with missing values.
 
    Parameters:
        data_dict (dict): Dictionary where keys are variable names, and values are 2D np.ndarray.
        axis (int): Axis along which to compute the mean:
                    - axis=0: Compute mean for each column (time).
                    - axis=1: Compute mean for each row (altitude).

    Returns:
        dict: Dictionary with imputed 2D arrays for numeric variables only.
    """
    imputed_data = []
    # Check if the data is a NumPy array and numeric
    if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
            # Compute the mean along the specified axis and replace NaNs
        mean_values = np.nanmean(data, axis=axis, keepdims=True)
        imputed_data = np.where(np.isnan(data) | (mask == 0), mean_values, data)
    else:
        print(f"Skipping non-numeric data ")
    return imputed_data

import numpy as np
from sklearn.impute import KNNImputer

def knn_imputation(data, k=3):
    """
    Perform traditional KNN imputation for missing values.

    Parameters:
        data (np.ndarray): Input data with missing values (NaN).
        k (int): Number of nearest neighbors to use for imputation.

    Returns:
        np.ndarray: Data with missing values imputed.
    """
    data = np.array(data, dtype=float)  # Ensure data is a float array
    n_rows, n_cols = data.shape
    
    # Iterate through each row
    for i in range(n_rows):
        missing_indices = np.where(np.isnan(data[i]))[0]  # Indices of missing values in the current row
        
        if len(missing_indices) == 0:
            continue  # Skip rows without missing values

        # Compute distances to other rows (excluding rows with NaNs in the same columns)
        distances = []
        for j in range(n_rows):
            if i == j:
                continue  # Skip the same row
            
            # Get indices of valid (non-NaN) columns for both rows
            valid_indices = ~np.isnan(data[i]) & ~np.isnan(data[j])
            
            if np.sum(valid_indices) == 0:
                distances.append(np.inf)  # Assign infinite distance if no valid indices
                continue
            
            # Compute Euclidean distance
            dist = np.linalg.norm(data[i, valid_indices] - data[j, valid_indices])
            distances.append(dist)
        
        # Sort distances and select k nearest neighbors
        neighbor_indices = np.argsort(distances)[:k]
        
        # Impute each missing value
        for idx in missing_indices:
            neighbor_values = []
            for neighbor in neighbor_indices:
                if not np.isnan(data[neighbor, idx]):
                    neighbor_values.append(data[neighbor, idx])
            
            # Compute the mean of neighbor values for imputation
            if neighbor_values:
                data[i, idx] = np.mean(neighbor_values)
    
    return data

def bayesian_pca_imputation(data, mask, n_components=2, iterations=1000):
    """
    Bayesian PCA for missing data imputation using scikit-learn.

    Parameters:
    - data: np.ndarray
        Input data matrix with missing values (use np.nan for missing entries).
    - mask: np.ndarray
        Boolean mask indicating missing values (True for missing, False for observed).
    - n_components: int
        Number of principal components to retain.
    - iterations: int
        Number of iterations for iterative PCA reconstruction.

    Returns:
    - imputed_data: np.ndarray
        Data with missing values imputed.
    """
    data = np.array(data, dtype=float)  # Ensure the data is a float ndarray
    mask = np.asarray(1- mask, dtype=bool)  # Ensure mask is boolean
    nan_mask = np.isnan(data) | mask  # Boolean mask for missing values
    data_imputed = np.copy(data)
    # Step 1: Replace NaNs with column means
    for i in range(data.shape[1]):
        col_mean = np.nanmean(data[:, i])
        data_imputed[nan_mask[:, i], i] = col_mean
    # Step 2: Standardize the data (ignoring NaNs)
    data_mean = np.nanmean(data_imputed, axis=0)
    data_std = np.nanstd(data_imputed, axis=0) + 1e-8  # Avoid division by zero
    standardized_data = (data_imputed - data_mean) / data_std
    # Initialize PCA
    pca = PCA(n_components=n_components)
    # Step 3: Perform iterative PCA
    for _ in range(iterations):
        # Fit PCA on rows without missing values
        if standardized_data.size == 0:
            raise ValueError("No valid rows without NaNs are available for PCA.")
        valid_data = np.nan_to_num(standardized_data, nan=0)
        pca.fit(valid_data)
        # Transform and reconstruct
        transformed_data = pca.transform(valid_data)
        reconstructed_data = pca.inverse_transform(transformed_data)
        # Update NaNs in the standardized data
        standardized_data[nan_mask] = reconstructed_data[nan_mask]
    # Step 4: Revert standardization
    reconstructed_data = standardized_data * data_std + data_mean
    imputed_data = np.copy(data)
    imputed_data[nan_mask] = reconstructed_data[nan_mask]
    return imputed_data

def interpolation_imputation(data, mask, method='linear', axis=0):
    """
    Interpolation-based imputation for missing data.

    Parameters:
    - data: np.ndarray
        Input data matrix with missing values (use np.nan for missing entries).
    - method: str
        Interpolation method. Options include 'linear', 'nearest', 'cubic', etc.
    - axis: int
        Axis along which to perform interpolation (0 for columns, 1 for rows).

    Returns:
    - imputed_data: np.ndarray
        Data with missing values imputed using interpolation.
    """
    # Ensure input data is a float array
    data = np.array(data, dtype=float)
    imputed_data = np.copy(data)
    mask = np.asarray(1-mask, dtype=bool)
    # Choose the axis to interpolate along
    if axis == 0:
        # Interpolate along columns
        for col in range(data.shape[1]):
            nan_mask = np.isnan(data[:, col]) | mask[:,col]
            if np.all(nan_mask):
                continue
            valid_idx = np.where(~nan_mask)[0]
            valid_data = data[valid_idx, col]
            if len(valid_idx) > 1:  # Ensure there are enough points to interpolate
                interpolator = interp1d(valid_idx, valid_data, kind=method, bounds_error=False, fill_value="extrapolate")
                imputed_data[nan_mask, col] = interpolator(np.where(nan_mask)[0])
    elif axis == 1:
        # Interpolate along rows
        for row in range(data.shape[0]):
            nan_mask = np.isnan(data[row, :]) | mask[row,:]
            if np.all(nan_mask):
                # If all values are NaN, skip interpolation
                continue
            valid_idx = np.where(~nan_mask)[0]
            valid_data = data[row, valid_idx]
            if len(valid_idx) > 1:  # Ensure there are enough points to interpolate
                interpolator = interp1d(valid_idx, valid_data, kind=method, bounds_error=False, fill_value="extrapolate")
                imputed_data[row, nan_mask] = interpolator(np.where(nan_mask)[0])
    else:
        raise ValueError("Axis must be 0 (columns) or 1 (rows).")

    return imputed_data





 
