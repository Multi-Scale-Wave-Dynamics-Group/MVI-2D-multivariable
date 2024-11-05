from metpy.units import units
import metpy.calc as mpcalc
import msise00
import numpy as np
import pandas as pd
import os 
# Constants
M_air = 28.97 * units.gram / units.mole  # Molar mass of dry air
R_specific = 287.05 * units('joule / kilogram / kelvin')  # Specific gas constant for dry air

def calculate_pressure(mass_density, temperature):
    """
    Calculate pressure using the ideal gas law, converting from particle density:
    P = (particle_density * particle_mass) * R * T
    
    Arguments:
        particle_density: Particle number density (in 1/m^3 with pint units).
        temperature: Temperature (in Kelvin with pint units).
        particle_mass: Mass of a single particle (in kg with pint units).
    
    Returns:
        Pressure in Pascals (with pint units).
    """
    R_specific = 287.05 * units('joule / kilogram / kelvin')  # Specific gas constant for dry air
    # Calculate pressure using the ideal gas law
    pressure = mass_density * R_specific * temperature
    return pressure  # Pressure in Pascals (Pa)

import pandas as pd
import numpy as np
from metpy import calc as mpcalc
from metpy.units import units

def calculate_potential_temperature(time, lat, lon, alt, actual_temp):
    """
    Retrieve atmospheric data using msise00 model, calculate pressure using the ideal gas law,
    and compute the potential temperature using MetPy. This function replicates each hour's
    atmospheric data 10 times to create a time x altitude map of shape (80, 71) and saves the 
    msise00 outputs and potential temperature as CSV files.
    """

    df,data_file = save_atmospheric_data(time, lat, lon, alt)
# Ensure temperature has proper units (Kelvin)
    temperature = actual_temp.flatten() * units.kelvin
    # Calculate total density with units
    total_density_with_units = df['TotMassDen'].values.flatten() * units.kg / (units.meter ** 3)
    # Calculate pressure in Pascals using the ideal gas law
    pressure = calculate_pressure(total_density_with_units, temperature)
    # Convert pressure to hPa for MetPy
    pressure_with_units = (pressure / 100.0).to(units.hPa)
    # Compute potential temperature
    potential_temperatures = mpcalc.potential_temperature(pressure_with_units, temperature).magnitude
    # Reshape potential temperatures to match time x altitude grid
    # Add potential temperatures to DataFrame and save to a new CSV file
    df['potential_temperature'] = potential_temperatures
    df.to_csv(data_file, index=False)
    print(f"Atmospheric and potential temperature data saved as {data_file}")
    return potential_temperatures

def calculate_density_ratio(time,lat,lon,alt,actual_number_density):
    df,_ = save_atmospheric_data(time, lat, lon, alt)
    actual_number_density_with_units = actual_number_density.flatten() / (units.centimeter ** 3) # from cm^3 to m^3
    total_number_density_with_units = df['TotNumDen'].values.flatten() / (units.meter ** 3)
    density_ratio = (actual_number_density_with_units * 10**6).to(1/units.meter ** 3) / total_number_density_with_units
    return density_ratio.magnitude


def save_atmospheric_data(time, lat, lon, alt):
    """
    Retrieve atmospheric data using the msise00 model, and save it as a CSV file.
    This function replicates each hour's atmospheric data 10 times to create a time x altitude 
    map of shape (80, 71) and saves the msise00 outputs directly without any further calculations.
    """
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    model_dir = base_dir + '/model/msise/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    file_name_msise_data = model_dir + f"{time[0].strftime('%Y%m%d_%H%M')}_msise00_atmospheric_data.csv"
    if os.path.isfile(file_name_msise_data):
        print(f"File {file_name_msise_data} exists. Loading atmospheric data.")
        msise_data_df = pd.read_csv(file_name_msise_data)
    else:
        print(f"File {file_name_msise_data} does not exist. Retrieving and saving atmospheric data.")
        msise_data_list = []
        unique_hours = sorted(set([dt.replace(minute=0, second=0, microsecond=0) for dt in time]))
        
        for hr in unique_hours:
            # Retrieve atmospheric data for this hour using MSISE00
            atmosphere = msise00.run(hr, alt, lat, lon)

            # Create a DataFrame for the retrieved atmosphere data
            atmosphere_df = pd.DataFrame({
                'He': atmosphere['He'].values.flatten(),
                'O': atmosphere['O'].values.flatten(),
                'N2': atmosphere['N2'].values.flatten(),
                'O2': atmosphere['O2'].values.flatten(),
                'Ar': atmosphere['Ar'].values.flatten(),
                'TotMassDen': atmosphere['Total'].values.flatten(),
                'Tn': atmosphere['Tn'].values.flatten(),
                'alt_km': atmosphere['alt_km'].values.flatten(),
                'lat': [lat] * len(atmosphere['alt_km']),
                'lon': [lon] * len(atmosphere['alt_km']),
                'hour': [hr] * len(atmosphere['alt_km'])
            })
            total_number_density = (atmosphere_df['He'].values + atmosphere_df['O'].values + atmosphere_df['N2'].values + atmosphere_df['O2'].values + atmosphere_df['Ar'].values)
            atmosphere_df['TotNumDen'] = total_number_density
            # Replicate atmospheric data for this hour 10 times
            atmosphere_replicated = pd.DataFrame(np.tile(atmosphere_df.values, (10, 1)), columns=atmosphere_df.columns)
            msise_data_list.append(atmosphere_replicated)
        
        # Concatenate all MSISE00 data into one DataFrame and save as CSV
        msise_data_df = pd.concat(msise_data_list, axis=0)
        msise_data_df.to_csv(file_name_msise_data, index=False)
        print(f"MSISE00 atmospheric data saved as {file_name_msise_data}")

    return msise_data_df,file_name_msise_data

