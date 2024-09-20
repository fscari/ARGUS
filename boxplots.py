import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the CSV file
file_path = r'C:/Users/localadmin/PycharmProjects/Argus/TTA_data/results_2024-09-20.csv'  # Replace <date> with the actual date or use a variable results_2024-09-17
data = pd.read_csv(file_path)

# Convert Power Control Status to categorical
data['Power Control Status'] = data['Power Control Status'].astype('category')

# Filter data for the specific fog percentages
fog_values = [0, 50, 100]

# Set up the plotting environment
plt.figure(figsize=(18, 10))

for i, fog in enumerate(fog_values, 1):
    plt.subplot(1, 3, i)
    fog_data = data[data['Fog Percentage'] == fog]
    sns.boxplot(x='Velocity', y='TTA', hue='Power Control Status', data=fog_data)
    plt.title(f'TTA Comparison at Fog = {fog}%')
    plt.xlabel('Velocity (km/h)')
    plt.ylabel('Time-to-Arrival (TTA)')
    plt.legend(title='Power Control Status')

# Adjust layout and show the plot
plt.tight_layout()
output_file = r'C:\Users\localadmin\PycharmProjects\Argus\TTA_data\results_2024-09-20.png'
plt.savefig(output_file)
plt.show()
