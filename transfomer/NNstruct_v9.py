### --- v9 --- ###
# Add multi-variables
# Add channel mixer
# testing code --- DataPrep_3rd.ipynb
import torch
import torch.nn as nn 
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.expanduser('/Users/huj7/Desktop/ERAU/MVI-2D/functions'))
from data_processing import process_files, load_daily_data

### --- Multivariable Model --- ###

class MultivariableLocalTransformerWithChannelMixer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, window_size, dropout=0.2):
        super(MultivariableLocalTransformerWithChannelMixer, self).__init__()
        
        # CNN block to capture small features from multivariable inputs
        self.cnn = nn.Sequential(
            nn.Conv2d(input_dim, d_model // 2, kernel_size=3, padding=1),  # First Conv Layer
            nn.ReLU(),
            nn.Conv2d(d_model // 2, d_model, kernel_size=3, padding=1),    # Second Conv Layer
            nn.ReLU(),
            nn.BatchNorm2d(d_model),  # Batch Normalization to stabilize training
        )
        
        # Channel mixer MLP to mix information across the channel (input_dim) axis
        self.channel_mixer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),  # First fully connected layer (expand dimension)
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),  # Second fully connected layer (project back)
        )

        # Initialize rotary embeddings for time and altitude
        self.rotary_time = RotaryEmbedding(dim=d_model // nhead)
        self.rotary_altitude = RotaryEmbedding(dim=d_model // nhead)

        # Transformer encoder layers for local windows
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Fully connected output layer to project back to input space (multivariable output)
        self.fc_out = nn.Linear(d_model, input_dim)  # Project back to input_dim (e.g., sodium density, wind, temperature, etc.)
        self.window_size = window_size

    def forward(self, src):
        # src shape: [batch_size, window_size_t, window_size_a, input_dim]
        batch_size, window_size_t, window_size_a, input_dim = src.shape

        # Apply CNN to capture small features in the time-altitude space
        src = src.permute(0, 3, 1, 2)  # Change to [batch_size, input_dim, window_size_t, window_size_a]
        src = self.cnn(src)            # Apply CNN: [batch_size, d_model, window_size_t, window_size_a]
        src = src.permute(0, 2, 3, 1)  # Change back to [batch_size, window_size_t, window_size_a, d_model]

        # Apply channel mixer MLP to mix information across the channel axis
        src = self.channel_mixer(src)  # Apply MLP: [batch_size, window_size_t, window_size_a, d_model]

        # Prepare rotary embeddings for time and altitude dimensions
        rotary_emb_time = self.rotary_time(torch.arange(window_size_t, device=src.device))
        rotary_emb_altitude = self.rotary_altitude(torch.arange(window_size_a, device=src.device))

        # Apply rotary embedding to time and altitude dimensions
        src_rotary_time = apply_rotary_emb(rotary_emb_time, src.transpose(1, 2)).transpose(1, 2)
        src_rotary = apply_rotary_emb(rotary_emb_altitude, src_rotary_time)

        # Flatten for transformer: [batch_size, window_size_t * window_size_a, d_model]
        src_rotary = src_rotary.view(batch_size, window_size_t * window_size_a, -1)

        # Pass through transformer
        transformer_out = self.transformer(src_rotary, src_rotary)

        # Reshape the output back to the original window shape
        transformer_out = transformer_out.reshape(batch_size, window_size_t, window_size_a, -1)

        # Final output: prediction for all variables (sodium density, wind, temperature, etc.)
        output = self.fc_out(transformer_out)

        return output

### --- Data Loading and Preprocessing --- ###
def load_multivariate_data(data_dir, files, start_UT, end_UT, variables):
    """
    Load multivariable data, including sodium density, wind, temperature, etc.
    Arguments:
        data_dir: Directory containing data files.
        files: List of filenames to load.
        start_UT, end_UT: Start and end time for extracting data.
        variables: List of variable names to load (e.g., ["Na Density", "Wind", "Temperature"]).
    Returns:
        Combined tensor of multivariable data and a fitted scaler for each variable.
    """
    all_data = []
    for filename in files:
        daily_data = load_daily_data(data_dir, [filename], start_UT, end_UT)
        combined = np.stack([np.transpose(np.stack(daily_data[var])) for var in variables], axis=-1)
        all_data.append(combined)

    # Stack all loaded files along the batch dimension
    all_data = np.stack(all_data)
    
    # Initialize a scaler for each variable
    scalers = {var: StandardScaler() for var in variables}
    # Normalize each variable independently
    for var_idx, var in enumerate(variables):
        # Reshape for scaling the variable independently (preserve batch and spatial dimensions)
        var_data = all_data[..., var_idx].reshape(-1, 1)  # Flatten along batch/time/altitude, keep var separate
        scaled_var_data = scalers[var].fit_transform(var_data).reshape(all_data.shape[:-1])  # Reshape back
        # Update the scaled data in the combined tensor
        all_data[..., var_idx] = scaled_var_data
    # Return the processed data and the scalers for each variable
    return torch.tensor(all_data, dtype=torch.float32), scalers

### --- Loss Functions --- ###
def dynamic_weighted_masked_loss(output, target, mask, base_weight):
    missing_count = torch.sum(1 - mask, dim=(1, 2))
    weight_adjusted = base_weight / (missing_count + 1e-6)
    reconstruction_loss = masked_loss(output, target, mask)
    weighted_loss = masked_loss(output, target, 1 - mask)
    total_loss = base_weight * weighted_loss + reconstruction_loss
    return total_loss

def masked_loss(output, target, mask):
    loss = nn.MSELoss(reduction='none')(output, target)
    loss = loss * mask
    return loss.mean()

def smoothness_loss(predictions):
    # Calculate the second-order gradient (laplacian) along time and altitude
    gradient_time = predictions[:, 2:, :, :] - 2 * predictions[:, 1:-1, :, :] + predictions[:, :-2, :, :]
    loss_time = torch.mean(torch.abs(gradient_time))
    return loss_time

### --- Physics Loss --- ###
def physics_loss(output, wind, time_steps, altitude_levels):
    """
    Compute the physics loss based on the equation: dρ/dt + w * dρ/dz = 0
    Arguments:
        output: Predicted data with shape [batch_size, time_steps, altitude_levels, num_variables]
        wind: Wind data with the same shape as output [batch_size, time_steps, altitude_levels]
        time_steps: Number of time steps in the data
        altitude_levels: Number of altitude levels in the data
    Returns:
        Physics loss based on finite differences
    """
    # Compute finite differences along the time axis (dρ/dt)
    d_rho_dt = output[:, 1:, :] - output[:, :-1, :]  # Shape: [batch_size, time_steps-1, altitude_levels, num_variables]
    
    # Compute finite differences along the altitude axis (dρ/dz)
    d_rho_dz = output[:, :, 1:] - output[:, :, :-1]  # Shape: [batch_size, time_steps, altitude_levels-1, num_variables]
    
    # Crop wind data to match the finite difference dimensions
    wind_cropped = wind[:, 1:, :-1]  # Shape: [batch_size, time_steps-1, altitude_levels-1, num_variables]
    
    # Physics constraint: dρ/dt + w * dρ/dz = 0
    physics_residual = d_rho_dt[:, :, :-1] + wind_cropped * d_rho_dz[:, :-1, :]
    
    # Compute the physics loss as the mean squared error of the residual
    physics_loss_value = torch.mean(physics_residual ** 2)
    
    return physics_loss_value

### --- Mask and local window --- ###
def mask_slices(data, mask_ratio, max_consecutive=3):

    batch_size, time_steps, altitude_levels, input_dim = data.shape  # Ensure the correct shape
    # Create a time mask for each batch and time step
    time_mask = np.random.rand(batch_size, time_steps) < mask_ratio
    
    # Ensure no more than `max_consecutive` time steps are masked consecutively
    #for i in range(batch_size):
    #    consecutive_count = 0
    #    for t in range(time_steps):
    #        if time_mask[i, t]:
    #            consecutive_count += 1
    #        else:
    #            consecutive_count = 0
    #        if consecutive_count > max_consecutive:
    #            time_mask[i, t] = False  # Unmask to break the streak of consecutive masked time steps
    #            consecutive_count = 0  # Reset the count after unmasking
    # Ensure no more than `max_consecutive` time steps are masked consecutively

    # Convert the mask to a torch tensor and move it to the same device as the data
    time_mask = torch.tensor(time_mask, dtype=torch.float32, device=data.device)
    
    # Expand the mask to match the altitude levels and input dimensions
    mask = time_mask.unsqueeze(2).unsqueeze(-1).expand(batch_size, time_steps, altitude_levels, input_dim)
    
    # If the mask has an extra singleton dimension, remove it with squeeze()
    mask = mask.squeeze(-1)
    
    # Apply the mask to the data
    masked_data = data * mask
    
    return masked_data, mask

def extract_local_window(data, mask, time_window_size, step_size):
    # Padding extra dimensions to the edge
    pad_size = time_window_size // 2

    # Ensure the mask does not have any extra dimensions
    if mask.shape[-1] == 1:
        mask = mask.squeeze(-1)  # Remove the extra dimension

    if data.shape != mask.shape:
        raise ValueError(f"Shape mismatch between data and mask: {data.shape} vs {mask.shape}")
    
    # Pad data and mask to ensure windows fit properly
    padded_data = torch.nn.functional.pad(data, (0, 0, 0, 0, pad_size, pad_size), mode='constant', value=0)
    padded_mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, pad_size, pad_size), mode='constant', value=0)

    # Extracting local regions with sliding windows using step_size
    local_windows, local_masks = [], []
    for b in range(padded_data.shape[0]):
        for t in range(0, padded_data.shape[1] - time_window_size + 1, step_size):
            local_window = padded_data[b, t:t + time_window_size, :, :]
            local_mask = padded_mask[b, t:t + time_window_size, :, :]
            if local_window.shape[0] == time_window_size and local_window.shape[1] == 71:
                local_windows.append(local_window)
                local_masks.append(local_mask)
    return torch.stack(local_windows), torch.stack(local_masks)

### --- Restore and Plot --- ###
def restore_output_to_original_shape(output, orig_batch_size, time_steps, altitude_levels, window_size, step_size):

    # Initialize an empty tensor to store the restored data
    restored_data = torch.zeros((orig_batch_size, time_steps, altitude_levels, output.shape[-1]), device=output.device)

    # Initialize a count matrix to track how many times each pixel has been predicted
    count_matrix = torch.zeros((orig_batch_size, time_steps, altitude_levels, 1), device=output.device)

    output_idx = 0
    for b in range(orig_batch_size):
        for t in range(0, time_steps - window_size + 1, step_size):
            # Extract the corresponding output window
            window_output = output[output_idx]

            # Ensure the window size matches
            if window_output.shape[0] == window_size:
                # Add the window output to the appropriate region of the restored data
                restored_data[b, t:t + window_size, :, :] += window_output

                # Increment the count matrix to keep track of the number of predictions for each pixel
                count_matrix[b, t:t + window_size, :, :] += 1

            output_idx += 1

    # Avoid division by zero: ensure all count_matrix elements are at least 1
    count_matrix[count_matrix == 0] = 1

    # Divide the restored data by the count matrix to average overlapping regions
    restored_data = restored_data / count_matrix

    return restored_data


def plot_and_save_2d_comparison(output, masked_data, batch_data, epoch, output_dir, time_steps, altitude_levels, index, variable_names,cat_str):
    # Remove batch dimension
    output = output.squeeze(0)  # Remove batch dimension (1, 80, 71, 3) -> (80, 71, 3)
    masked_data = masked_data.squeeze(0)  # Remove batch dimension
    batch_data = batch_data.squeeze(0)  # Remove batch dimension

    num_variables = output.shape[-1]  # Number of variables (e.g., 3 for sodium density, wind, temperature)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for var_idx in range(num_variables):
        var_name = variable_names[var_idx]  # Name of the variable (e.g., "Sodium Density", "Wind", "Temperature")

        # Detach data and move to CPU without clipping
        output_var = output[..., var_idx].detach().cpu().numpy()
        masked_data_var = masked_data[..., var_idx].detach().cpu().numpy()
        batch_data_var = batch_data[..., var_idx].detach().cpu().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Plot the predicted output for this variable
        im0 = axs[0].imshow(output_var, aspect='auto', cmap='viridis', extent=[0, altitude_levels, 0, time_steps])
        axs[0].set_title(f'Predicted {var_name} (Epoch {epoch})')
        axs[0].set_xlabel('Altitude Levels')
        axs[0].set_ylabel('Time Steps')
        fig.colorbar(im0, ax=axs[0])

        # Plot the masked data for this variable
        im1 = axs[1].imshow(masked_data_var, aspect='auto', cmap='viridis', extent=[0, altitude_levels, 0, time_steps])
        axs[1].set_title(f'Masked {var_name} (Epoch {epoch})')
        axs[1].set_xlabel('Altitude Levels')
        axs[1].set_ylabel('Time Steps')
        fig.colorbar(im1, ax=axs[1])

        # Plot the original data for this variable
        im2 = axs[2].imshow(batch_data_var, aspect='auto', cmap='viridis', extent=[0, altitude_levels, 0, time_steps])
        axs[2].set_title(f'Original {var_name} (Epoch {epoch})')
        axs[2].set_xlabel('Altitude Levels')
        axs[2].set_ylabel('Time Steps')
        fig.colorbar(im2, ax=axs[2])

        # Save the figure for each variable
        file_name = f'epoch_{epoch}_file_{index}_var_{var_idx}_{cat_str}.png'
        plt.savefig(os.path.join(output_dir, file_name))
        plt.close()


### --- Training Function --- ###
def train_model_multivariate(
    model, 
    train_files, 
    val_files, 
    data_dir, 
    variables, 
    optimizer, 
    num_epochs, 
    device, 
    batch_size, 
    start_UT, 
    end_UT, 
    output_dir, 
    window_size, 
    mask_ratio, 
    time_steps, 
    altitude_levels, 
    step_size, 
    base_weight, 
    lambda_smooth
):

    best_loss = float('inf')
    epoch_losses = []  # List to store training loss for each epoch
    val_losses = []  # List to store validation loss for each epoch

    for epoch in range(num_epochs):
        print(f"\nStarting epoch {epoch + 1}/{num_epochs}")

        ### Training Phase ###
        model.train()
        total_train_loss = 0

        for index, filename in enumerate(train_files):
            # Load training data and scalers
            train_data, scalers = load_multivariate_data(data_dir, [filename], start_UT, end_UT, variables)

            # Mask creation for training
            mask_train = torch.isnan(train_data).float().to(device)
            mask_train = 1.0 - mask_train  # NaNs to 0
            train_data = torch.nan_to_num(train_data, nan=0.0).to(device)
            target_data = train_data.clone()

            # Create DataLoader
            train_dataset = TensorDataset(train_data, target_data, mask_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            for batch_idx, (batch_data, batch_target, batch_mask) in enumerate(train_loader):
                batch_data, batch_target, batch_mask = batch_data.to(device), batch_target.to(device), batch_mask.to(device)

                optimizer.zero_grad()

                # Forward pass
                masked_data, random_mask = mask_slices(batch_data, mask_ratio=mask_ratio)
                random_mask = random_mask.unsqueeze(-1)
                local_windows, local_masks = extract_local_window(masked_data, random_mask, window_size, step_size)
                local_windows_orig, local_masks_orig = extract_local_window(batch_data, batch_mask, window_size, step_size)
                output = model(local_windows)

                # Restore output to original shape
                restored_output = restore_output_to_original_shape(output, batch_data.shape[0], time_steps, altitude_levels, window_size, step_size)

                # Plot and save the comparison images for training
                plot_and_save_2d_comparison(restored_output, masked_data, batch_data, epoch, output_dir, time_steps, altitude_levels, batch_idx, variables,'train')

                # Loss calculation
                combined_mask = local_masks * local_masks_orig
                mse_loss = dynamic_weighted_masked_loss(output, local_windows_orig, combined_mask, base_weight)
                smooth_loss = smoothness_loss(output)
                #total_batch_loss = mse_loss + lambda_smooth * smooth_loss

                #
                # Physics loss computation
                wind_data = batch_data[..., 1]  # Assuming wind is the second variable in the input
                physics_loss_value = physics_loss(restored_output[...,0], wind_data, time_steps, altitude_levels)
                # Total loss
                total_batch_loss = mse_loss + lambda_smooth * smooth_loss + 1 * physics_loss_value

                total_batch_loss.backward()
                optimizer.step()

                total_train_loss += total_batch_loss.item()

        avg_train_loss = total_train_loss / len(train_files)
        epoch_losses.append(avg_train_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}')

         # Save the model if it improves
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            print(f"Epoch {epoch + 1}: Improved validation loss = {avg_train_loss}. Saving model...")

            model_save_path = os.path.join(output_dir, f'model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at: {model_save_path}")
            
        ### Validation Phase ###
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for index, filename in enumerate(val_files):
                val_data, scalers = load_multivariate_data(data_dir, [filename], start_UT, end_UT, variables)
                mask_val = torch.isnan(val_data).float().to(device)
                mask_val = 1.0 - mask_val
                val_data = torch.nan_to_num(val_data, nan=0.0).to(device)
                target_val_data = val_data.clone()

                val_dataset = TensorDataset(val_data, target_val_data, mask_val)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                for batch_idx, (batch_data, batch_target, batch_mask) in enumerate(val_loader):
                    batch_data, batch_target, batch_mask = batch_data.to(device), batch_target.to(device), batch_mask.to(device)

                    masked_data = batch_data * batch_mask
                    local_windows, local_masks = extract_local_window(masked_data, batch_mask, window_size, step_size)
                    local_windows_orig, local_masks_orig = extract_local_window(batch_data, batch_mask, window_size, step_size)
                    output = model(local_windows)

                    restored_output = restore_output_to_original_shape(output, batch_data.shape[0], time_steps, altitude_levels, window_size, step_size)

                    plot_and_save_2d_comparison(restored_output, masked_data, batch_data, epoch, output_dir, time_steps, altitude_levels, batch_idx, variables,'test')

                    combined_mask = local_masks * local_masks_orig
                    val_loss = dynamic_weighted_masked_loss(output, local_windows_orig, combined_mask, base_weight)

                    smooth_loss_val = smoothness_loss(output)
                    total_val_loss += val_loss.item() + lambda_smooth * smooth_loss_val.item()

        avg_val_loss = total_val_loss / len(val_files)
        val_losses.append(avg_val_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss}')
