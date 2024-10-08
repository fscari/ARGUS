import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the lidar data from the CSV file
file_name = "Lidar_data_exp_15 - Copy"
file_path = fr'C:/Users/localadmin/PycharmProjects/Argus/lidar_data/{file_name}.csv'  # Replace <date> with the actual date or use a variable results_2024-09-17
print('reading LiDAR file')
lidar_data = pd.read_csv(file_path)
print('LiDAR file read')

# Load the TTA data from the CSV file
tta_file_name = file_name.replace('Lidar_data', 'tta')
tta_file_path = fr'C:/Users/localadmin/PycharmProjects/Argus/TTA_data/{tta_file_name}.csv'
print('reading TTA file')
tta_data = pd.read_csv(tta_file_path)
print('TTA file read')

# compute angle of the points from the x and y columns
lidar_data['point_angle'] = np.degrees(np.arctan2(lidar_data['y'], lidar_data['x']))
# check if the angle is bigger than the angle in the driver_angle column
in_central_vision = np.abs(lidar_data['point_angle'] - lidar_data['driver_angle']) <= 30

lidar_data['in_ROI'] = lidar_data['point_angle'] < lidar_data['driver_angle'] - 30
# count how many points are in the ROI for each iteration_nr
print(lidar_data.groupby('iteration_nr')['in_ROI'].sum())
# devide the number of points in the ROI by the total number of points to get the percentage of points in the ROI per iteration number
print(lidar_data.groupby('iteration_nr')['in_ROI'].sum() / lidar_data.groupby('iteration_nr')['in_ROI'].count())

# create a new df with the number of points in the ROI for each iteration
df = lidar_data.groupby('iteration_nr')['in_ROI'].sum().reset_index()
# add the fog value, the power control status and the frequency control status to the new df based on the iteration number from the tta_data
df['Fog Percentage'] = tta_data['Fog Percentage']
df['Power Control Status'] = tta_data['Power Control Status']
df['Frequency Control Status'] = tta_data['Frequency Control Status']
df['Velocity'] = tta_data['Velocity']

# check if even iteration numbers have more points in the ROI than odd iteration number before
for i in range(1, len(df), 2):
    if df.loc[i, 'in_ROI'] < df.loc[i-1, 'in_ROI']:
        print(f'Even iteration number {i} has less points in the ROI than odd iteration number {i-1}')

# compute average number of points in the ROI for each velocity and frequency control status
print(df.groupby(['Velocity', 'Frequency Control Status'])['in_ROI'].mean())

# Filter data for the specific fog percentages
fog_values = [0, 50, 100]

# Set up the plotting environment
plt.figure(figsize=(18, 10))

for i, fog in enumerate(fog_values, 1):
    plt.subplot(1, 3, i)
    fog_data = df[df['Fog Percentage'] == fog]
    sns.boxplot(x='Velocity', y='in_ROI', hue='Frequency Control Status', data=fog_data)
    plt.title(f'Number of data points Comparison at Fog = {fog}%')
    plt.xlabel('Velocity (km/h)')
    plt.ylabel('Number of points in the ROI')
    plt.legend(title='Frequency Control Status')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
