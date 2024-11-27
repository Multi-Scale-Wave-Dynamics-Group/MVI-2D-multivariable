import matplotlib.pyplot as plt
import numpy as np

def plot_file_data(all_imputed_data_dict, all_mask_dict, all_original_data_dict, save_plots=False, output_dir="plots"):
    """
    Plot three subplots for each file: imputed, masked, and original data.

    Parameters:
        all_imputed_data_dict (dict): Dictionary of imputed data.
        all_mask_dict (dict): Dictionary of masks used during imputation.
        all_original_data_dict (dict): Dictionary of original data.
        save_plots (bool): Whether to save the plots as images.
        output_dir (str): Directory to save the plots (if save_plots is True).
    """
    for key in all_imputed_data_dict.keys():
        imputed_datasets = all_imputed_data_dict[key]
        original_datasets = all_original_data_dict[key]
        masked_datasets = all_mask_dict[key]

        # Iterate through each dataset in the current variable
        for idx, (imputed_data, original_data, mask) in enumerate(zip(imputed_datasets, original_datasets, masked_datasets)):
            print(f"Plotting for {key} - File {idx + 1}")
            
            fig, axes = plt.subplots(1, 2, figsize=(8, 6), constrained_layout=True)

            # Plot 1: Imputed data
            ax = axes[0]
            if imputed_data.ndim == 2:
                im = ax.imshow(imputed_data-original_data, aspect='auto', cmap='viridis')
                fig.colorbar(im, ax=ax)
            else:
                ax.plot(imputed_data-original_data)
            ax.set_title("Imputed Data")
            ax.set_ylabel("Altitude" if imputed_data.ndim == 2 else "Value")
            ax.set_xlabel("Time")

            # Plot 3: Original data
            ax = axes[1]
            if original_data.ndim == 2:
                im = ax.imshow(mask , aspect='auto', cmap='viridis')
                fig.colorbar(im, ax=ax)
            else:
                ax.plot(original_data)
            ax.set_title("Original Data")
            ax.set_ylabel("Altitude" if original_data.ndim == 2 else "Value")
            ax.set_xlabel("Time")

            # Save the plot if requested
            if save_plots:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                plot_path = os.path.join(output_dir, f"{key.replace(' ', '_')}_file_{idx + 1}.png")
                plt.savefig(plot_path)
                print(f"Saved plot to {plot_path}")
            
            # Show the plot
            plt.show()