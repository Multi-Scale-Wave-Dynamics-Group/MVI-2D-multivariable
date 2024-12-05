import matplotlib.pyplot as plt
from datetime import datetime, timedelta
def plot_imputation_results(
    original_data, masked_data, restored_data, scalers, variables, time_labels, altitude_labels, 
    font_config,  # Pass the font configuration dictionary
    time_step=20, altitude_step=20, rotation=45
):
    num_vars = len(variables)
    rows, cols = 3, num_vars
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True)
    date_label = time_labels[0][:8]
    formatted_date = datetime.strptime(date_label, "%Y%m%d").strftime("%m-%d-%Y")
    fig.suptitle(f"Imputation Results for Date: {formatted_date}", fontsize=font_config["title_fontsize"], y=1.02)

    reduced_time_labels = [label.split('_')[1][:2] for label in time_labels[::time_step]]
    reduced_time_indices = range(0, len(time_labels), time_step)
    reduced_altitude_labels = altitude_labels[::altitude_step]
    reduced_altitude_indices = range(0, len(altitude_labels), altitude_step)

    variable_units = {
        "Na Density (cm^(-3))": r"$\text{Na Density (cm}^{-3}\text{)}$",
        "Vertical Wind (m/s)": r"$\text{Vertical Wind (m/s)}$",
        "Temperature (K)": r"$\text{Temperature (K)}$"
    }

    for row in range(rows):
        for col in range(cols):
            var = variables[col]
            final_output_rescaled = scalers[var].inverse_transform(
                restored_data[..., col].reshape(-1, 1)
            ).reshape(original_data.shape[0], original_data.shape[1])
            imputed_diff = final_output_rescaled - original_data[..., col]

            if row == 0:
                im = axs[row, col].imshow(final_output_rescaled, aspect='auto', cmap='viridis', origin='lower')
                axs[row, col].set_title(f"{variable_units[var]} \n(Imputed - Original)", fontsize=font_config["title_fontsize"])
            elif row == 1:
                im = axs[row, col].imshow(masked_data[..., col], aspect='auto', cmap='viridis', origin='lower')
                axs[row, col].set_title("(Masked Data)", fontsize=font_config["title_fontsize"])
            elif row == 2:
                im = axs[row, col].imshow(original_data[..., col], aspect='auto', cmap='viridis', origin='lower')
                axs[row, col].set_title("(Original Data)", fontsize=font_config["title_fontsize"])

            if row == rows - 1:
                axs[row, col].set_xlabel("Altitude (km)", fontsize=font_config["label_fontsize"])
                axs[row, col].set_xticks(reduced_altitude_indices)
                axs[row, col].set_xticklabels(reduced_altitude_labels, fontsize=font_config["tick_label_fontsize"], rotation=rotation)
            else:
                axs[row, col].set_xticks([])

            if col == 0:
                axs[row, col].set_ylabel("Universal Hour", fontsize=font_config["label_fontsize"])
                axs[row, col].set_yticks(reduced_time_indices)
                axs[row, col].set_yticklabels(reduced_time_labels, fontsize=font_config["tick_label_fontsize"], rotation=-rotation)
            else:
                axs[row, col].set_yticks([])

            fig.colorbar(im, ax=axs[row, col], orientation='vertical', pad=0.02)
    plt.show()
