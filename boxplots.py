import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the CSV file
file_path = r'C:/Users/localadmin/PycharmProjects/Argus/TTA_data/results_2024-09-17.csv'  # Replace <date> with the actual date or use a variable
data = pd.read_csv(file_path)
# Convert Power Control Status to categorical
data['Power Control Status'] = data['Power Control Status'].astype('category')
# Set up the plotting environment
plt.figure(figsize=(14, 12))

# First plot: Boxplot for each fog density
plt.subplot(2, 1, 1)
sns.boxplot(x='Fog Percentage', y='TTA', hue='Power Control Status', data=data)
plt.title('TTA Comparison by Fog Density and Power Control Status')
plt.xlabel('Fog Percentage')
plt.ylabel('Time-to-Arrival (TTA)')
plt.legend(title='Power Control Status')

# Second plot: Boxplot combining all fog densities together
plt.subplot(2, 1, 2)
sns.boxplot(x='Power Control Status', y='TTA', data=data)
plt.title('TTA Comparison Across All Fog Densities')
plt.xlabel('Power Control Status')
plt.ylabel('Time-to-Arrival (TTA)')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()