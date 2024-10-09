import numpy as np
import pandas as pd
import os


def save_density(experiment_nr, fog_density, density_directory, lidar_directory, tta_directory):
    lidar_file_name = fr'Lidar_data_exp_{experiment_nr}_fog_density_{fog_density}'
    lidar_file_path = os.path.join(lidar_directory, f'{lidar_file_name}.csv')
    print('reading LiDAR file')
    lidar_data = pd.read_csv(lidar_file_path)
    print('LiDAR file read')

    # Load the TTA data from the CSV file
    tta_file_name = fr'tta_exp_{experiment_nr}'
    tta_file_path = os.path.join(tta_directory, f'{tta_file_name}.csv')
    print('reading TTA file')
    tta_data = pd.read_csv(tta_file_path)
    print('TTA file read')

    # compute angle of the points from the x and y columns
    lidar_data['point_angle'] = np.degrees(np.arctan2(lidar_data['y'], lidar_data['x']))
    # check if the angle is bigger than the angle in the driver_angle column
    lidar_data['in_ROI'] = lidar_data['point_angle'] < lidar_data['driver_angle'] - 30

    # Save the density data
    density_file_name = fr"density_data_exp_{experiment_nr}_fog_{fog_density}"
    density_file_path = os.path.join(density_directory, f'{density_file_name}.csv')

    df = lidar_data.groupby('iteration_nr')['in_ROI'].sum().reset_index()
    # add the fog value, the power control status and the frequency control status to the new df based on the iteration number from the tta_data
    df = df.merge(tta_data[['iteration_nr', 'Fog Percentage', 'Power Control Status', 'Frequency Control Status', 'Velocity']], on='iteration_nr')

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


# Test the function
if __name__ == "__main__":
    experiment_nr = 2
    fog_density = 100
    exp_directory = fr'C:\Users\localadmin\PycharmProjects\Argus\Experiments\Experiment_{experiment_nr}'
    density_directory = os.path.join(exp_directory, 'Density')
    lidar_directory = os.path.join(exp_directory, 'LiDAR')
    tta_directory = os.path.join(exp_directory, 'TTA')
    save_density(experiment_nr, fog_density, density_directory, lidar_directory, tta_directory)
