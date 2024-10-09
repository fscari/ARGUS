import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def density_plot(density_directory, experiment_nr):
    fog_values = [0, 50, 100]
    # concatenate the data from the different fog values
    df = pd.DataFrame()
    for fog in fog_values:
        density_file_name = fr"density_data_exp_{experiment_nr}_fog_{fog}"
        density_file_path = os.path.join(density_directory, f'{density_file_name}.csv')
        if os.path.exists(density_file_path):
            density_data = pd.read_csv(density_file_path)
            df = pd.concat([df, density_data])

    # save the data to the file
    density_file_name = fr"density_data_exp_{experiment_nr}"
    density_file_path = os.path.join(density_directory, f'{density_file_name}.csv')
    df.to_csv(density_file_path, index=False)

    global_min = 0
    global_max = df['in_ROI'].max() + 100000

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
        # Set the same y-axis limits for all subplots
        plt.ylim(global_min, global_max)

    # Adjust layout and show the plot
    plt.tight_layout()
    # change csv to png in density_file_path
    output_file = density_file_path.replace('.csv', '.png')
    plt.savefig(output_file)
    # plt.show()

# Test the function
if __name__ == "__main__":
    experiment_nr = 2
    exp_directory = fr'C:\Users\localadmin\PycharmProjects\Argus\Experiments\Experiment_{experiment_nr}'
    density_directory = os.path.join(exp_directory, 'Density')
    density_plot(density_directory, experiment_nr)
