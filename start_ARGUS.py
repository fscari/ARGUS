import os
import csv
import sys
import time
import globals
from user_input import user_input
from save_density import save_density
from density_plot import density_plot
from boxplots import boxplots
from ARGUS import ARGUS


def start_ARGUS(iteration_nr_joan=None):
    if iteration_nr_joan is None and len(sys.argv) > 1:
        iteration_nr_joan = sys.argv[1]
        iteration_nr_joan = int(iteration_nr_joan)
    print("Starting a new ARGUS experiment...")
    exps_directory = r'C:\Users\localadmin\PycharmProjects\Argus\Experiments'
    folders = os.listdir(exps_directory)
    experiment_numbers = [int(folder.split('_')[-1]) for folder in folders if folder.startswith('Experiment')]
    last_experiment_nr = max(experiment_numbers) if experiment_numbers else 1
    next_experiment_nr = int(last_experiment_nr + 1)
    experiment_nr, fog_density, iteration_nr, control_type = user_input(iteration_nr_joan, next_experiment_nr)
    print(f"Experiment Number: {experiment_nr}")
    if not iteration_nr:
        iteration_nr = int(iteration_nr_joan)
    if isinstance(fog_density, list):
        print(f"Fog Densities: {fog_density}")
        iteration_nr = int(iteration_nr / len(fog_density))
    else:
        print(f"Fog Density: {fog_density}")
    print(f"Iteration Number: {iteration_nr}")
    print(f"Control Type: {control_type}")


    # create the experiment folder
    exp_directory = os.path.join(exps_directory, f'Experiment_{experiment_nr}')
    if not os.path.exists(exp_directory):
        os.makedirs(exp_directory)
        print(f"Experiment folder created: {exp_directory}")
    # create folders in the experiment folder
    tta_directory = os.path.join(exp_directory, 'TTA')
    if not os.path.exists(tta_directory):
        os.makedirs(tta_directory)
        print(f"TTA folder created: {tta_directory}")
    lidar_directory = os.path.join(exp_directory, 'LiDAR')
    if not os.path.exists(lidar_directory):
        os.makedirs(lidar_directory)
        print(f"LiDAR folder created: {lidar_directory}")
    density_directory = os.path.join(exp_directory, 'Density')
    if not os.path.exists(density_directory):
        os.makedirs(density_directory)
        print(f"Density folder created: {density_directory}")

    tta_file_name = fr'tta_exp_{experiment_nr}.csv'
    tta_file_path = os.path.join(tta_directory, tta_file_name)
    if not os.path.exists(tta_file_path):
        with open(tta_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["iteration_nr", "Fog Percentage", "Velocity", "Power Control Status", "Frequency Control Status", "TTA"])
    freq = False
    count = 0
    for fog_density in fog_density:
        for i in range(iteration_nr):
            globals.reset_globals()
            time.sleep(1)
            if count % 2 == 0:
                power_control = False
                drivers_gaze = False
                freq = False
                print(f"Condition ln")
            else:
                if control_type == 'power_control':
                    power_control = True
                    drivers_gaze = False
                    print(f"Condition lpic")
                elif control_type == 'frequency_control':
                    power_control = False
                    drivers_gaze = True
                    print(f"Condition lpfc")
                elif control_type == 'pf':
                    power_control = True
                    drivers_gaze = True
                    print(f"Condition lpifc")
                else:
                    freq = True
                    pass
            print(f'Power control active: {power_control}')
            print(f'Frequency control active: {drivers_gaze}')
            with open('/Users/localadmin/PycharmProjects/Argus/status_file.txt', 'w') as f:
                f.write("Experiment completed")
            ARGUS(tta_file_path, fog_density, count, experiment_nr, lidar_directory, power_control=power_control, drivers_gaze=drivers_gaze, freq=freq)
            count += 1
            print(f"Experiment {count} completed")
        print(f"Experiment fog density {fog_density} completed")
        save_density(experiment_nr, fog_density, density_directory, lidar_directory, tta_directory)
        print(f"Density data saved")
    density_plot(density_directory, experiment_nr)
    boxplots(tta_file_path)
    print("All experiments completed")

if __name__ == "__main__":
    iteration_nr_joan = sys.argv[1] if len(sys.argv) > 1 else None
    start_ARGUS(iteration_nr_joan)