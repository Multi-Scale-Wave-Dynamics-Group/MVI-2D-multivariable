import numpy as np
from datetime import datetime, timedelta
import os
import re

import numpy as np

def mask_slices(data, missing_ratio, strip_width):
    """
    Artificially crop missing time strips from 2D time series data with random strip widths.

    Parameters:
        data (numpy.ndarray): The 2D time series data [time x altitude].
        missing_ratio (float): Ratio of total time steps to crop as missing strips.
        max_strip_width (int): Maximum width of a missing strip.

    Returns:
        numpy.ndarray: Data with missing time strips.
        numpy.ndarray: Mask indicating valid (True) and missing (False) entries.
    """
    n_time, n_altitude = data.shape

    # Calculate the approximate number of total time steps to be masked
    total_missing_time_steps = int(missing_ratio * n_time)

    # Initialize the mask as all True (no missing data)
    mask = np.ones_like(data, dtype=bool)

    # Track the number of time steps already masked
    masked_steps = 0

    while masked_steps < total_missing_time_steps:
        # Randomly select a strip width between 1 and max_strip_width
        current_strip_width = np.random.randint(1, strip_width + 1)

        # Randomly select a start index for the strip
        start_index = np.random.randint(0, n_time - current_strip_width + 1)

        # If this strip will exceed the desired missing steps, adjust its width
        current_strip_width = min(current_strip_width, total_missing_time_steps - masked_steps)

        # Apply the mask for the selected strip
        mask[start_index:start_index + current_strip_width, :] = False

        # Update the number of masked steps
        masked_steps += current_strip_width

    # Apply the mask to create missing strips (set missing data to NaN)
    cropped_data = np.where(mask, data, np.nan)

    return cropped_data, mask


def convert_to_date_number(date_str, ut_hours):
    """Convert date string and UT hours to formatted date numbers."""
    base_date = datetime.strptime(date_str, '%Y%m%d')
    date_numbers = []
    for hour in ut_hours:
        delta = timedelta(hours=hour)
        new_date = base_date + delta
        date_numbers.append(new_date.strftime('%Y%m%d_%H%M'))
    return np.array(date_numbers)
# Column and row definitions
col = "UT (hour) for each column"
row = "Altitudes (km) for each row"
mat_list = [
    "Na Density (cm^(-3))",
    "Na Density Error (cm^(-3))",
    "Temperature (K)",
    "Temperature Error (K)",
    "Vertical Wind (m/s)",
    "Vertical Wind Error (m/s)",
    "Zonal Wind (m/s)",
    "Zonal Wind Error (m/s)",
    "Meridional Wind (m/s)",
    "Meridional Wind Error (m/s)"
]

def load_daily_data(filename, start_UT, end_UT):
    """Load daily data from a file and interpolate for missing time steps."""
    mat_flag = False
    col_flag = False
    row_flag = False
    col_np = None
    row_np = None
    mat_np = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line == col:
                col_flag = True
                continue
            if line == row:
                row_flag = True
                continue
            if line in mat_list:
                mat_flag = line
                continue
            if line == '':
                continue
            if col_flag:
                col_np = np.fromstring(line, dtype=float, sep=' ')
                col_flag = False
            elif row_flag:
                row_np = np.fromstring(line, dtype=float, sep=' ')
                row_flag = False
            elif mat_flag:
                if mat_flag in mat_np:
                    mat_np[mat_flag] = np.vstack([mat_np[mat_flag], np.fromstring(line, dtype=float, sep=' ')])
                else:
                    mat_np[mat_flag] = np.fromstring(line, dtype=float, sep=' ')
    if col_np is None or row_np is None or not mat_np:
        raise ValueError("Input file is missing required data or improperly formatted.")
    full_hours = np.arange(start_UT, end_UT, 0.1)  # Define full hours array
    full_mat_np = {}
    if np.min(col_np) <= start_UT or np.max(col_np) > end_UT:
        for key in mat_np.keys():
            mat_np[key][mat_np[key] == -999.0] = np.nan
            full_mat_np[key] = np.full((mat_np[key].shape[0], len(full_hours)), np.nan)  # Initialize with NaN
            valid_indices = (col_np > start_UT) & (col_np < end_UT)
            valid_col_np = col_np[valid_indices]
            indices = np.searchsorted(full_hours, valid_col_np)
            valid_mat_np = mat_np[key][:, valid_indices]  # Filter corresponding data
            full_mat_np[key][:, indices] = valid_mat_np
        datestr_match = re.match(r"^\d{8}", os.path.basename(filename))
        if not datestr_match:
            raise ValueError(f"Filename {filename} does not contain a valid date in 'YYYYMMDD' format.")
        datestr = datestr_match.group()
        datenum = convert_to_date_number(datestr, full_hours)  # Use full_hours for date numbers
        full_mat_np['YYYYMMDD_hhmm'] = datenum
        full_mat_np['Altitudes [km]'] = row_np
    else:
        full_mat_np = {}  # Return empty if data does not cover the time range
    return full_mat_np