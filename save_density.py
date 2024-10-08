import numpy as np
import pandas as pd
import os


def save_density(experimnet_nr):
    lidar_file_name = fr'Lidar_data_exp_{experimnet_nr}'
    file_path = fr'C:/Users/localadmin/PycharmProjects/Argus/lidar_data/{lidar_file_name}.csv'  # Replace <date> with the actual date or use a variable results_2024-09-17
    print('reading LiDAR file')
    lidar_data = pd.read_csv(file_path)
    print('LiDAR file read')

    # Load the TTA data from the CSV file
    tta_file_name = lidar_file_name.replace('Lidar_data', 'tta')
    tta_file_path = fr'C:/Users/localadmin/PycharmProjects/Argus/TTA_data/{tta_file_name}.csv'
    print('reading TTA file')
    tta_data = pd.read_csv(tta_file_path)
    print('TTA file read')

    # compute angle of the points from the x and y columns
    lidar_data['point_angle'] = np.degrees(np.arctan2(lidar_data['y'], lidar_data['x']))
    # check if the angle is bigger than the angle in the driver_angle column
    lidar_data['in_ROI'] = lidar_data['point_angle'] < lidar_data['driver_angle'] - 30

    # Save the density data
    density_file_name = fr"density_data_exp_{experimnet_nr}"
    density_file_path = fr'C:/Users/localadmin/PycharmProjects/Argus/lidar_data/{density_file_name}.csv'

    df = lidar_data.groupby('iteration_nr')['in_ROI'].sum().reset_index()
    # add the fog value, the power control status and the frequency control status to the new df based on the iteration number from the tta_data
    df['Fog Percentage'] = tta_data['Fog Percentage']
    df['Power Control Status'] = tta_data['Power Control Status']
    df['Frequency Control Status'] = tta_data['Frequency Control Status']
    df['Velocity'] = tta_data['Velocity']

    if os.path.exists(density_file_path):
        # open the file and read the data
        print('reading density file')
        density_data = pd.read_csv(density_file_path)
        print('density file read')
        # concatenate the new data to the existing data
        density_data = pd.concat([density_data, df])
        # save the data to the file
        density_data.to_csv(density_file_path, index=False)
    else:
        print('density file does not exist')
        # save the data to the file
        df.to_csv(density_file_path, index=False)
        # create a new df with the number of points in the ROI for each iteration
