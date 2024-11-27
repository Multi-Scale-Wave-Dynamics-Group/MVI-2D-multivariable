import os
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import re
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

def convert_to_date_number(date_str, ut_hours):
    base_date = datetime.strptime(date_str, '%Y%m%d')
    date_numbers = []
    for hour in ut_hours:
        delta = timedelta(hours=hour)
        new_date = base_date + delta
        date_numbers.append(new_date.strftime('%Y%m%d_%H%M'))
    return np.array(date_numbers)
    
def process_files(files, start_UT, end_UT, data_dir):
    stacked_data = {}

    for filename in files:
        mat_flag = False
        col_flag = False
        row_flag = False
        col_np = []
        row_np = []
        mat_np = {}

        with open(os.path.join(data_dir, filename), 'r') as f:
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
                if row_flag:
                    row_np = np.fromstring(line, dtype=float, sep=' ')
                    row_flag = False
                if mat_flag:
                    if mat_flag in mat_np.keys():
                        mat_np[mat_flag] = np.vstack([mat_np[mat_flag], np.fromstring(line, dtype=float, sep=' ')])
                    else:
                        mat_np[mat_flag] = np.fromstring(line, dtype=float, sep=' ')

        full_hours = np.arange(start_UT, end_UT, 0.1)  # Uniform time range from 0 to 12 hours with 0.1 hour resolution
        full_mat_np = {}
        for key in mat_np.keys():
            mat_np[key][mat_np[key] == -999.0] = np.nan
            full_mat_np[key] = np.empty((mat_np[key].shape[0], len(full_hours)))
            full_mat_np[key][:] = np.nan  # Initialize with NaN
            valid_indices = (col_np >= start_UT) & (col_np < end_UT)  # Filter valid indices within the range 0 to 12
            valid_col_np = col_np[valid_indices]
            indices = np.searchsorted(full_hours, valid_col_np)
            valid_mat_np = mat_np[key][:, valid_indices]  # Filter corresponding data
            full_mat_np[key][:, indices] = valid_mat_np

        for key in full_mat_np.keys():
            if key not in stacked_data:
                stacked_data[key] = full_mat_np[key]
            else:
                stacked_data[key] = np.hstack([stacked_data[key], full_mat_np[key]])

        datestr = filename.split('_')[0]
        datenum = convert_to_date_number(datestr, full_hours)  # Use full_hours for date numbers

        if 'YYYYMMDD_hhmm' not in stacked_data.keys():
            stacked_data['YYYYMMDD_hhmm'] = datenum
        else:
            stacked_data['YYYYMMDD_hhmm'] = np.hstack([stacked_data['YYYYMMDD_hhmm'], datenum])

    stacked_data['Altitudes [km]'] = row_np

    return stacked_data

def load_daily_data(filename,start_UT,end_UT):
    mat_flag = False
    col_flag = False
    row_flag = False
    col_np = []
    row_np = []
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
            if row_flag:
                row_np = np.fromstring(line, dtype=float, sep=' ')
                row_flag = False
            if mat_flag:
                if mat_flag in mat_np.keys():
                    mat_np[mat_flag] = np.vstack([mat_np[mat_flag], np.fromstring(line, dtype=float, sep=' ')])
                else:
                    mat_np[mat_flag] = np.fromstring(line, dtype=float, sep=' ')
    full_hours = np.arange(start_UT, end_UT, 0.1)  
    full_mat_np = {}
    for key in mat_np.keys():
        mat_np[key][mat_np[key] == -999.0] = np.nan
        full_mat_np[key] = np.empty((mat_np[key].shape[0], len(full_hours)))
        full_mat_np[key][:] = np.nan  # Initialize with NaN
        valid_indices = (col_np > start_UT) & (col_np < end_UT)  
        valid_col_np = col_np[valid_indices]
        indices = np.searchsorted(full_hours, valid_col_np)
        valid_mat_np = mat_np[key][:, valid_indices]  # Filter corresponding data
        full_mat_np[key][:, indices] = valid_mat_np
    datestr = re.match(r"^\d{8}", os.path.basename(filename)).group()
    datenum = convert_to_date_number(datestr, full_hours)  # Use full_hours for date numbers
    full_mat_np['YYYYMMDD_hhmm'] = datenum
    full_mat_np['Altitudes [km]'] = row_np
    return full_mat_np

def create_missing_strips_for_dict(data_dict, num_strips, strip_width):
    num_time_points = next(iter(data_dict.values())).shape[0]
    
    missing_indices = []
    for _ in range(num_strips):
        start_idx = np.random.randint(0, num_time_points - strip_width)
        end_idx = start_idx + strip_width
        missing_indices.append((start_idx, end_idx))

    data_with_missing = {}
    for key, data in data_dict.items():
        if key != 'YYYYMMDD_hhmm' and key != 'Altitudes [km]':
            data_copy = data.copy()
            for start_idx, end_idx in missing_indices:
                data_copy[:,start_idx:end_idx] = np.nan
            data_with_missing[key] = data_copy
        elif key in ['YYYYMMDD_hhmm', 'Altitudes [km]']:
            data_with_missing[key] = np.copy(data)

    return data_with_missing

def fill_missing_data(data):
  filled_data = {}
  filled_data['YYYYMMDD_hhmm'] = data['YYYYMMDD_hhmm']
  filled_data['Altitudes [km]'] = data['Altitudes [km]']
  for key in data.keys():
    if key != 'YYYYMMDD_hhmm' and key != 'Altitudes [km]':
      filled_data[key] = np.copy(data[key])
      for i in range(filled_data[key].shape[0]):
        row = filled_data[key][i, :]
        valid_indices = ~np.isnan(row)
        if np.sum(valid_indices) > 1:  # Need at least two valid points for interpolation
          f = interp1d(np.arange(len(row))[valid_indices],row[valid_indices],kind='nearest', fill_value="extrapolate")
          filled_data[key][i, :] = f(np.arange(len(row)))
  return filled_data