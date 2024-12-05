### --- v10 --- ###
# Updated at: Nov. 4 2024
# Add multi-variables
# Add channel mixer
# Testing code --- DataPrep_3rd.ipynb
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from metpy.units import units
import matplotlib.pyplot as plt
from datetime import datetime

# List customed functions
home_path = os.path.dirname(os.getcwd())
function_path = os.path.join(home_path,'functions')
sys.path.append(os.path.expanduser(function_path))
from data_processing import load_daily_data
from P_var_update import calculate_potential_temperature, calculate_density_ratio

### --- Multivariable Model --- ###
class MultivariableLocalTransformerWithChannelMixer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, window_size, dropout=0.2):
        super(MultivariableLocalTransformerWithChannelMixer, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_dim, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(d_model),
        )
        self.channel_mixer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.rotary_time = RotaryEmbedding(dim=d_model // nhead)
        self.rotary_altitude = RotaryEmbedding(dim=d_model // nhead)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, input_dim)
        self.window_size = window_size

    def forward(self, src):
        batch_size, window_size_t, window_size_a, input_dim = src.shape
        src = src.permute(0, 3, 1, 2)
        src = self.cnn(src)
        src = src.permute(0, 2, 3, 1)
        src = self.channel_mixer(src)
        rotary_emb_time = self.rotary_time(torch.arange(window_size_t, device=src.device))
        rotary_emb_altitude = self.rotary_altitude(torch.arange(window_size_a, device=src.device))
        src_rotary_time = apply_rotary_emb(rotary_emb_time, src.transpose(1, 2)).transpose(1, 2)
        src_rotary = apply_rotary_emb(rotary_emb_altitude, src_rotary_time)
        src_rotary = src_rotary.view(batch_size, window_size_t * window_size_a, -1)
        transformer_out = self.transformer(src_rotary, src_rotary)
        transformer_out = transformer_out.reshape(batch_size, window_size_t, window_size_a, -1)
        output = self.fc_out(transformer_out)
        return output

### --- Data Loading and Preprocessing --- ###
def load_multivariate_data(files, start_UT, end_UT, variables):
    all_data, all_times, all_altitudes = [], [], []
    for filename in files:
        daily_data = load_daily_data(filename, start_UT, end_UT)
        time_data = daily_data['YYYYMMDD_hhmm']
        datetime_list = [datetime.strptime(t, '%Y%m%d_%H%M') for t in time_data]
        altitude_data = daily_data['Altitudes [km]']
        all_times.append(datetime_list)
        all_altitudes.append(altitude_data)
        combined = np.stack([np.transpose(np.stack(daily_data[var])) for var in variables], axis=-1)
        all_data.append(combined)
    all_data = np.stack(all_data)
    all_times = np.stack(all_times)
    all_altitudes = np.stack(all_altitudes)
    scalers = {var: StandardScaler() for var in variables}
    for var_idx, var in enumerate(variables):
        var_data = all_data[..., var_idx].reshape(-1, 1)
        scaled_var_data = scalers[var].fit_transform(var_data).reshape(all_data.shape[:-1])
        all_data[..., var_idx] = scaled_var_data
    return torch.tensor(all_data, dtype=torch.float32), all_times, all_altitudes, scalers

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
    gradient_time = predictions[:, 2:, :, :] - 2 * predictions[:, 1:-1, :, :] + predictions[:, :-2, :, :]
    loss_time = torch.mean(torch.abs(gradient_time))
    return loss_time

### --- Physics Loss --- ###
def physics_loss(density, wind, time, lat, lon, alt, scaler_density, scaler_wind):
    den_unnormalized = scaler_density.inverse_transform(density.detach().cpu().numpy().reshape(-1, 1)).reshape(density.shape)
    wind_unnormalized = scaler_wind.inverse_transform(wind.detach().cpu().numpy().reshape(-1, 1)).reshape(wind.shape)
    density_ratio = calculate_density_ratio(time, lat, lon, alt, den_unnormalized).reshape(density.shape)
    d_rho_dt = density_ratio[:, 1:, :] - density_ratio[:, :-1, :]
    d_rho_dz = density_ratio[:, :, 1:] - density_ratio[:, :, :-1]
    wind_cropped = wind_unnormalized[:, 1:, :-1]
    physics_residual = d_rho_dt[:, :, :-1] + wind_cropped * d_rho_dz[:, :-1, :]
    return np.mean(physics_residual ** 2)

def physics_loss_lnT(actual_temp, wind, time, lat, lon, alt, scaler_temperature, scaler_wind):
    temp_unnormalized = scaler_temperature.inverse_transform(actual_temp.detach().cpu().numpy().reshape(-1, 1)).reshape(actual_temp.shape)
    wind_unnormalized = scaler_wind.inverse_transform(wind.detach().cpu().numpy().reshape(-1, 1)).reshape(wind.shape)
    potential_temp = calculate_potential_temperature(time, lat, lon, alt, temp_unnormalized)
    lnT = np.log(temp_unnormalized)
    dlnT_dt = np.gradient(lnT, axis=1)
    alt_meters = alt * 1000
    alt_meters = np.broadcast_to(alt_meters, (temp_unnormalized.shape[0], len(alt_meters)))
    dtheta_dz = np.gradient(potential_temp.reshape(1, 80, 71), np.squeeze(alt_meters), axis=-1)
    wind_pint = wind_unnormalized * units.meter / units.second
    physics_residual = dlnT_dt.flatten() + (wind_pint.flatten() * dtheta_dz.flatten() / potential_temp).magnitude
    return np.mean(physics_residual ** 2)

### --- Masking and Local Window Extraction --- ###
def mask_slices(data, mask_ratio, max_strip_width):
    batch_size, time_steps, altitude_levels, input_dim = data.shape
    total_missing_time_steps = int(mask_ratio * batch_size * time_steps)
    mask = np.ones((batch_size, time_steps, altitude_levels), dtype=bool)
    masked_steps = 0
    while masked_steps < total_missing_time_steps:
        current_strip_width = np.random.randint(1, max_strip_width + 1)
        batch_index = np.random.randint(0, batch_size)
        start_index = np.random.randint(0, time_steps - current_strip_width + 1)
        current_strip_width = min(current_strip_width, total_missing_time_steps - masked_steps)
        mask[batch_index, start_index:start_index + current_strip_width, :] = False
        masked_steps += current_strip_width
    mask = torch.tensor(mask, dtype=torch.float32, device=data.device).unsqueeze(-1).expand(batch_size, time_steps, altitude_levels, input_dim)
    masked_data = data * mask
    return masked_data, mask

def extract_local_window(data, mask, time_window_size, step_size):
    pad_size = time_window_size // 2
    if mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if data.shape != mask.shape:
        raise ValueError(f"Shape mismatch between data and mask: {data.shape} vs {mask.shape}")
    padded_data = torch.nn.functional.pad(data, (0, 0, 0, 0, pad_size, pad_size), mode='constant', value=0)
    padded_mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, pad_size, pad_size), mode='constant', value=0)
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
    restored_data = torch.zeros((orig_batch_size, time_steps, altitude_levels, output.shape[-1]), device=output.device)
    count_matrix = torch.zeros((orig_batch_size, time_steps, altitude_levels, 1), device=output.device)
    output_idx = 0
    for b in range(orig_batch_size):
        for t in range(0, time_steps - window_size + 1, step_size):
            window_output = output[output_idx]
            if window_output.shape[0] == window_size:
                restored_data[b, t:t + window_size, :, :] += window_output
                count_matrix[b, t:t + window_size, :, :] += 1
            output_idx += 1
    count_matrix[count_matrix == 0] = 1
    restored_data = restored_data / count_matrix
    return restored_data

def plot_and_save_2d_comparison(output, masked_data, batch_data, epoch, output_dir, time_steps, altitude_levels, index, variable_names, cat_str):
    output = output.squeeze(0)
    masked_data = masked_data.squeeze(0)
    batch_data = batch_data.squeeze(0)
    num_variables = output.shape[-1]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for var_idx in range(num_variables):
        var_name = variable_names[var_idx]
        output_var = output[..., var_idx].detach().cpu().numpy()
        masked_data_var = masked_data[..., var_idx].detach().cpu().numpy()
        batch_data_var = batch_data[..., var_idx].detach().cpu().numpy()
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        im0 = axs[0].imshow(output_var, aspect='auto', cmap='viridis', extent=[0, altitude_levels, 0, time_steps])
        axs[0].set_title(f'Predicted {var_name} (Epoch {epoch})')
        fig.colorbar(im0, ax=axs[0])
        im1 = axs[1].imshow(masked_data_var, aspect='auto', cmap='viridis', extent=[0, altitude_levels, 0, time_steps])
        axs[1].set_title(f'Masked {var_name} (Epoch {epoch})')
        fig.colorbar(im1, ax=axs[1])
        im2 = axs[2].imshow(batch_data_var, aspect='auto', cmap='viridis', extent=[0, altitude_levels, 0, time_steps])
        axs[2].set_title(f'Original {var_name} (Epoch {epoch})')
        fig.colorbar(im2, ax=axs[2])
        file_name = f'epoch_{epoch}_file_{index}_var_{var_idx}_{cat_str}.png'
        plt.savefig(os.path.join(output_dir, file_name))
        plt.close()

### --- Training Function --- ###
def train_model_multivariate(
    model, train_files, val_files, data_dir, variables, optimizer, num_epochs, device,
    batch_size, start_UT, end_UT, output_dir, window_size, mask_ratio, max_strip_width,
    time_steps, altitude_levels, step_size, base_weight, lambda_smooth, physics_loss_flag
):
    lat, lon = -31.17, -70.81
    best_loss = float('inf')
    total_loss = 0
    epoch_losses, val_losses = [], []
    for epoch in range(num_epochs):
        print(f"\nStarting epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_batch_loss = 0
        for index, filename in enumerate(train_files):
            train_data, datetime_list, altitudes, scalers = load_multivariate_data([filename], start_UT, end_UT, variables)
            scaler_temp = scalers["Temperature (K)"]
            scaler_wind = scalers["Vertical Wind (m/s)"]
            scaler_density = scalers["Na Density (cm^(-3))"]
            mask_train = torch.isnan(train_data).float().to(device)
            mask_train = 1.0 - mask_train
            train_data = torch.nan_to_num(train_data, nan=0.0).to(device)
            target_data = train_data.clone()
            train_loader = DataLoader(TensorDataset(train_data, target_data, mask_train), batch_size=batch_size, shuffle=True)
            for batch_idx, (batch_data, batch_target, batch_mask) in enumerate(train_loader):
                torch.mps.empty_cache()
                batch_data, batch_target, batch_mask = batch_data.to(device), batch_target.to(device), batch_mask.to(device)
                optimizer.zero_grad()
                masked_data, random_mask = mask_slices(batch_data, mask_ratio, max_strip_width)
                random_mask = random_mask.unsqueeze(-1)
                local_windows, local_masks = extract_local_window(masked_data, random_mask, window_size, step_size)
                local_windows_orig, local_masks_orig = extract_local_window(batch_data, batch_mask, window_size, step_size)
                output = model(local_windows)
                restored_output = restore_output_to_original_shape(output, batch_data.shape[0], time_steps, altitude_levels, window_size, step_size)
                plot_and_save_2d_comparison(restored_output, masked_data, batch_data, epoch, output_dir, time_steps, altitude_levels, index, variables, 'train')
                combined_mask = local_masks * local_masks_orig
                mse_loss = dynamic_weighted_masked_loss(output, local_windows_orig, combined_mask, base_weight)
                smooth_loss = smoothness_loss(output)
                total_batch_loss = mse_loss + lambda_smooth * smooth_loss
                if physics_loss_flag:
                    physics_loss_wind = physics_loss(restored_output[..., 0], batch_data[..., 1], np.squeeze(datetime_list), lat, lon, np.squeeze(altitudes), scaler_density, scaler_wind)
                    temp_data = restored_output[..., 2]
                    wind_data = restored_output[..., 1]
                    physics_loss_temp = physics_loss_lnT(temp_data, wind_data, np.squeeze(datetime_list), lat, lon, np.squeeze(altitudes), scaler_temp, scaler_wind)
                    total_batch_loss = mse_loss + lambda_smooth * smooth_loss + physics_loss_wind
                    if not np.isnan(physics_loss_temp):
                        total_batch_loss += physics_loss_temp
                total_batch_loss.backward()
                total_loss += total_batch_loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()  # Perform weight update
        avg_train_loss = total_loss / len(train_files)
        epoch_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}")
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            model_save_path = os.path.join(output_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at: {model_save_path}")
        ### Validation Phase ###
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for index, filename in enumerate(val_files):
                val_data,_,_,_ = load_multivariate_data([filename], start_UT, end_UT, variables)
                mask_val = torch.isnan(val_data).float().to(device)
                mask_val = 1.0 - mask_val
                val_data = torch.nan_to_num(val_data, nan=0.0).to(device)
                target_val_data = val_data.clone()
                val_dataset = TensorDataset(val_data, target_val_data, mask_val)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                for batch_idx,(batch_data, batch_target, batch_mask) in enumerate(val_loader):
                    batch_data, batch_target, batch_mask = batch_data.to(device), batch_target.to(device), batch_mask.to(device)
                    masked_data = batch_data * batch_mask
                    masked_data, random_mask = mask_slices(batch_data, mask_ratio, max_strip_width)
                    random_mask = random_mask.unsqueeze(-1)
                    local_windows, local_masks = extract_local_window(masked_data, batch_mask, window_size, step_size)
                    local_windows_orig, local_masks_orig = extract_local_window(batch_data, batch_mask, window_size, step_size)
                    output = model(local_windows)
                    restored_output = restore_output_to_original_shape(output, batch_data.shape[0], time_steps, altitude_levels, window_size, step_size)
                    plot_and_save_2d_comparison(restored_output, masked_data, batch_data, epoch, output_dir, time_steps, altitude_levels, index, variables,'test')
                    combined_mask = local_masks * local_masks_orig
                    val_loss = dynamic_weighted_masked_loss(output, local_windows_orig, combined_mask, base_weight)
                    smooth_loss_val = smoothness_loss(output)
                    if physics_loss_flag == True:
                        physics_loss_wind = physics_loss(restored_output[..., 0], batch_data[..., 1], np.squeeze(datetime_list), lat, lon, np.squeeze(altitudes), scaler_density, scaler_wind)
                        temp_data = restored_output[..., 2]
                        wind_data = restored_output[..., 1]
                        physics_loss_temp = physics_loss_lnT(temp_data, wind_data, np.squeeze(datetime_list), lat, lon, np.squeeze(altitudes), scaler_temp, scaler_wind)
                        total_loss = mse_loss + lambda_smooth * smooth_loss + physics_loss_wind
                        if not np.isnan(physics_loss_temp):
                            total_loss += physics_loss_temp
                    elif physics_loss_flag == False:
                        total_loss = mse_loss + lambda_smooth * smooth_loss
                        total_val_loss += val_loss.item() + lambda_smooth * smooth_loss_val.item()

        avg_val_loss = total_val_loss / len(val_files)
        val_losses.append(avg_val_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss}')


