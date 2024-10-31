import msise00
import datetime
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units  # Use MetPy's unit registry
import os
import pandas as pd
from metpy.units import units


#output dir
output_dir = '/Users/huj7/Desktop/ERAU/MVI-2D-multivariable/'

from metpy.units import units
import numpy as np

# Constants
M_air = 28.97 * units.gram / units.mole  # Molar mass of dry air
R_specific = 287.05 * units('joule / kilogram / kelvin')  # Specific gas constant for dry air

def calculate_pressure(total_density, temperature):
    """
    Calculate pressure using the ideal gas law: P = rho * R * T
    where:
    - rho is the total density in kg/m^3
    - R is the specific gas constant (J/kg/K)
    - T is the temperature in Kelvin
    
    Arguments:
        total_density: Total density of gases (in kg/m^3 with pint units).
        temperature: Temperature (in Kelvin with pint units).
    
    Returns:
        Pressure in Pascals (with pint units).
    """
    # Ensure total_density has units of kg/m^3 and temperature in Kelvin
    total_density_with_units = total_density

    # Calculate pressure using the ideal gas law
    pressure = total_density_with_units * R_specific * temperature

    return pressure  # Pressure will be in Pascals (Pa)



# Function to batch process MSISE00 calls
# Assume `calculate_potential_temperature` function exists
import metpy.calc as mpcalc
import numpy as np
from metpy.units import units

def calculate_potential_temperature(time, lat, lon, alt):
    """
    Retrieve atmospheric data using msise00 model, calculate pressure using the ideal gas law,
    and compute the potential temperature using MetPy. This function replicates each hour's
    atmospheric data 10 times to create a time x altitude map of shape (80, 71).
    """
    potential_temperatures = []
    
    # Get unique hours first
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
            'Tn': atmosphere['Tn'].values.flatten(),
            'alt_km': atmosphere['alt_km'].values.flatten(),
            'lat': [lat] * len(atmosphere['alt_km']),
            'lon': [lon] * len(atmosphere['alt_km'])
        })

        # Ensure temperature has proper units (Kelvin)
        temperature = atmosphere_df['Tn'].values * units.kelvin
        
        # Total density calculation (assuming all components are in kg/m^3)
        total_density = (atmosphere_df['He'].values + atmosphere_df['O'].values + atmosphere_df['N2'].values + atmosphere_df['O2'].values + atmosphere_df['Ar'].values)
        total_density_with_units = total_density * units.kg / (units.meter ** 3)
        # Calculate pressure in Pascals using the ideal gas law
        # total_density_with_units [kg/m^3]
        # temperature[K]
        pressure = calculate_pressure(total_density_with_units, temperature)

        # Convert pressure from Pascals to hPa for use in MetPy
        pressure_with_units = (pressure / 100.0).to(units.hPa)

        # Now compute potential temperature using MetPy (pressure in hPa, temperature in Kelvin)
        potential_temp = mpcalc.potential_temperature(pressure_with_units, temperature).magnitude

        # Replicate the potential temperature for this hour 10 times
        potential_temp_replicated = np.tile(potential_temp, (10, 1))  # Shape becomes (10, 71)

        # Append the replicated temperatures for this hour
        potential_temperatures.append(potential_temp_replicated)

    # Stack all hours' results into a single array with shape (80, 71)
    potential_temperatures = np.vstack(potential_temperatures)  # Shape becomes (80, 71)

    return potential_temperatures.flatten()

# Assuming the function `calculate_pressure` is defined elsewhere
def calculate_buoyancy_frequency_sqrd(time, lat, lon, alt):
    """
    Calculate the buoyancy frequency (N) using the MSISE00 atmospheric model.
    N = sqrt(g / theta * dtheta/dz)
    Arguments:
        time: The time of the observation (datetime).
        lat: Latitude of the observation (degrees).
        lon: Longitude of the observation (degrees).
        alt: Altitude levels (in kilometers).
    Returns:
        Buoyancy frequency N (in radians/second).
    """
    # Constants
    g = 9.81 * units.meter / units.second**2  # Gravitational acceleration
    # Retrieve atmospheric data and calculate potential temperature
    potential_temp = calculate_potential_temperature(time, lat, lon, alt)
    mesh_time,mesh_alt = np.meshgrid(time,alt)
    alt = mesh_alt.flatten()
    # Compute the vertical gradient of potential temperature dtheta/dz
    alt_meters = alt * 1000 * units.meter  # Convert altitude from km to meters
    dtheta_dz = np.gradient(potential_temp, alt_meters)  # Vertical gradient of potential temperature
    
    # Compute buoyancy frequency N
    buoyancy_freq_sqrd = (g / potential_temp) * dtheta_dz  # Exclude the last point to match dimensions
    return buoyancy_freq_sqrd

