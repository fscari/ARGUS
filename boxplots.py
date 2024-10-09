import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def boxplots(tta_file_path):
    # Load the TTA data from the CSV file
    data = pd.read_csv(tta_file_path)

    # Convert Power Control Status to categorical
    data['Frequency Control Status'] = data['Frequency Control Status'].astype('category') # Frequency Control Status , Power Control Status

    # Filter data for the specific fog percentages
    fog_values = [0, 50, 100]

    # Set up the plotting environment
    plt.figure(figsize=(18, 10))

    for i, fog in enumerate(fog_values, 1):
        plt.subplot(1, 3, i)
        fog_data = data[data['Fog Percentage'] == fog]
        sns.boxplot(x='Velocity', y='TTA', hue='Frequency Control Status', data=fog_data)
        plt.title(f'TTA Comparison at Fog = {fog}%')
        plt.xlabel('Velocity (km/h)')
        plt.ylabel('Time-to-Arrival (TTA)')
        plt.legend(title='Frequency Control Status')

    # Adjust layout and show the plot
    plt.tight_layout()
    output_file = tta_file_path.replace('.csv', '.png')
    plt.savefig(output_file)
    # plt.show()


# Test the function
if __name__ == "__main__":
    experiment_nr = 2
    tta_file_path = fr'C:\Users\localadmin\PycharmProjects\Argus\Experiments\Experiment_{experiment_nr}\TTA\tta_exp_{experiment_nr}.csv'
    boxplots(tta_file_path, experiment_nr)
