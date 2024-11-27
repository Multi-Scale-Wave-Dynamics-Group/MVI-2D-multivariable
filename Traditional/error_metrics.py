from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Function to calculate metrics
def calculate_metrics(all_imputed_data_dict, all_original_data_dict, all_mask_dict):
    metrics_dict = {}
    for key in all_imputed_data_dict.keys():

        imputed_data_list = all_imputed_data_dict[key]
        original_data_list = all_original_data_dict[key]
        mask_list = all_mask_dict[key]
        # Flatten the arrays for each variable
        all_imputed = np.concatenate([data.flatten() for data in imputed_data_list])
        all_original = np.concatenate([data.flatten() for data in original_data_list])
        all_mask = np.concatenate([mask.flatten() for mask in mask_list])
        # Apply the mask to select only imputed positions
        masked_imputed = all_imputed[all_mask == 0]
        masked_original = all_original[all_mask == 0]
        # Remove NaNs from both masked_imputed and masked_original
        valid_indices = ~np.isnan(masked_imputed) & ~np.isnan(masked_original)
        masked_imputed = masked_imputed[valid_indices]
        masked_original = masked_original[valid_indices]
        # Calculate metrics
        mae = np.mean(np.abs(masked_imputed - masked_original))
        mse = mean_squared_error(masked_original, masked_imputed)
        rmse = np.sqrt(mse)
        re = np.mean(np.abs((masked_imputed - masked_original) /masked_original))
        r2 = r2_score(masked_original, masked_imputed)
        metrics_dict[key] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "re": re,
            "R^2": r2,
            "Mean Imputed": np.mean(masked_imputed),
            "Std Dev Imputed": np.std(masked_imputed)
        }
    return metrics_dict
