import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the CSV file
file_name = "tta_exp_15"
file_path = fr'C:/Users/localadmin/PycharmProjects/Argus/TTA_data/{file_name}.csv'  # Replace <date> with the actual date or use a variable results_2024-09-17
data = pd.read_csv(file_path)

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
output_file = fr'C:\Users\localadmin\PycharmProjects\Argus\TTA_data\{file_name}.png'
plt.savefig(output_file)
plt.show()
