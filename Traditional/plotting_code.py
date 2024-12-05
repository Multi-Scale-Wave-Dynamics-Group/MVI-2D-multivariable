import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

def plot_traditional_imputation_results(
    all_original_data_dict, 
    all_mask_dict, 
    all_imputed_data_dict, 
    time_labels, 
    altitude_labels, 
    time_step=20, 
    altitude_step=20, 
    fontsize=20, 
    labelsize=18,
    rotation=45
):
    """
    Plot traditional imputation results for all files in the dataset.

    Parameters:
        all_original_data_dict: dict
            Dictionary containing original data (86 x 120 x 39 for each variable).
        all_mask_dict: dict
            Dictionary containing masked data (86 x 120 x 39 for each variable).
        all_imputed_data_dict: dict
            Dictionary containing imputed data (86 x 120 x 39 for each variable).
        time_labels: list
            List of time labels (e.g., "Hour_1").
        altitude_labels: list
            List of altitude labels (e.g., "80", "81").
        fontsize: int
            Font size for plot titles and labels.
        labelsize: int
            Font size for axis labels.
        time_step: int
            Step size for reducing the number of time labels.
        altitude_step: int
            Step size for reducing the number of altitude labels.
        rotation: int
            Rotation angle for x-axis labels.
    """
    variables = list(all_original_data_dict.keys())
    rows, cols = 3, len(variables)

    num_files = np.array(all_original_data_dict[variables[0]]).shape[0]
    
    # Reduced labels for cleaner visualization
    reduced_time_labels = [label.split('_')[1][:2] for label in time_labels[::time_step]]
    reduced_time_indices = list(range(0, len(time_labels), time_step))
    reduced_altitude_labels = altitude_labels[::altitude_step]
    reduced_altitude_indices = list(range(0, len(altitude_labels), altitude_step))

    variable_units = {
        "Na Density (cm^(-3))": r"$\text{Na Density (cm}^{-3}\text{)}$",
        "Vertical Wind (m/s)": r"$\text{Vertical Wind (m/s)}$",
        "Temperature (K)": r"$\text{Temperature (K)}$"
    }

    for file_idx in range(num_files):
    # Set up the figure for each file
        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True)
        date_label = time_labels[0][:8]
        formatted_date = datetime.strptime(date_label, "%Y%m%d").strftime("%m-%d-%Y")  # Convert to 'MM-DD-YYYY'
        fig.suptitle(f"Imputation Results for Date: {formatted_date}", fontsize=fontsize + 4, y=1.02)
        for row in range(rows):
            for col in range(cols):
                var = variables[col]
                original_data = all_original_data_dict[var][file_idx]
                masked_data = all_mask_dict[var][file_idx]
                imputed_data = all_imputed_data_dict[var][file_idx]
                imputed_diff = imputed_data - original_data

            # Select the correct data for each subplot
                if row == 0:  # Imputed - Original
                    im = axs[row, col].imshow(imputed_diff, aspect='auto', cmap='seismic', origin='lower')
                    axs[row, col].set_title(f"{variable_units[var]} \n(Imputed - Original)", fontsize=fontsize + 2)
                elif row == 1:  # Masked Data
                    im = axs[row, col].imshow(masked_data * original_data, aspect='auto', cmap='viridis', origin='lower')
                    axs[row, col].set_title(f"{variable_units[var]} \n(Masked Data)", fontsize=fontsize + 2)
                elif row == 2:  # Original Data
                    im = axs[row, col].imshow(original_data, aspect='auto', cmap='viridis', origin='lower')
                    axs[row, col].set_title(f"{variable_units[var]} \n(Original Data)", fontsize=fontsize + 2)
            # Configure axis labels
                if row == rows - 1:  # Only bottom row gets x-axis labels
                    axs[row, col].set_xlabel("Altitude (km)", fontsize=fontsize)
                    axs[row, col].set_xticks(reduced_altitude_indices)
                    axs[row, col].set_xticklabels(reduced_altitude_labels, fontsize=labelsize, rotation=rotation)
                else:
                    axs[row, col].set_xticks([])

                if col == 0:  # Only leftmost column gets y-axis labels
                    axs[row, col].set_ylabel("Universal Hour", fontsize=fontsize)
                    axs[row, col].set_yticks(reduced_time_indices)
                    axs[row, col].set_yticklabels(reduced_time_labels, fontsize=labelsize)
                else:
                    axs[row, col].set_yticks([])

                # Add colorbar
                fig.colorbar(im, ax=axs[row, col], orientation='vertical', pad=0.02)

    # Display the figure
        plt.show()