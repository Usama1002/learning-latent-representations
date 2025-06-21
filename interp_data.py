import pandas as pd
import os
import numpy as np

file_path = 'data/original_data/1_ISI.csv'
data = pd.read_csv(file_path, header=0, index_col=0)
data = pd.DataFrame(data)

# Use the first 'TIME ' column as the reference for interpolation
timestep = data['TIME '].dropna()
list_timestep = timestep.to_list()

interval = 10**(-12)
num_sample = int((list_timestep[-1] - list_timestep[0]) // interval + 1)
new_timestep = [list_timestep[0] + interval * i for i in range(num_sample)]

# Prepare new DataFrame with all columns, inserting NaNs for missing timesteps
new_data = data.copy()
rows_to_add = []
for step in new_timestep:
    if step not in list_timestep:
        # Insert a row with NaNs except for 'TIME '
        row = {col: np.nan for col in data.columns}
        row['TIME '] = step
        rows_to_add.append(row)
if rows_to_add:
    new_data = pd.concat([new_data, pd.DataFrame(rows_to_add)], ignore_index=True)

# Sort by 'TIME ' and set up timestamp for time interpolation
new_data = new_data.sort_values(by=['TIME '])
new_data['timestamp'] = new_data['TIME '].apply(lambda x: pd.Timestamp(x * 1000, unit='s'))
new_data.set_index(['timestamp'], inplace=True)

# Interpolate all columns except the index
cols_to_interp = [col for col in new_data.columns if col != 'timestamp']
new_data[cols_to_interp] = new_data[cols_to_interp].interpolate(method='time', limit_direction='both')

# Restore the original order and drop the timestamp index
new_data.reset_index(drop=True, inplace=True)
new_data = new_data[data.columns]  # Ensure same column order as original

# Save to new location with same structure
os.makedirs('data/interp_data', exist_ok=True)
interp_file_path = os.path.join('data/interp_data', os.path.basename(file_path))
new_data.to_csv(interp_file_path)

print(f"Interpolated data saved to {interp_file_path}")
