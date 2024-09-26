import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d


def read_lidar_data(file_path):
    # Load the data from the CSV file
    df = pd.read_csv(file_path)

    print(f"Data loaded from {file_path}")
    print(f"Total points: {len(df)}")

    # Display the first few rows of the dataframe
    print("\nFirst few rows of data:")
    print(df.head())

    return df


def visualize_lidar_data(df):
    # Create a point cloud object in Open3D
    pcd = o3d.geometry.PointCloud()

    # Set the point cloud data from the DataFrame
    points = df[['x', 'y', 'z']].values
    colors = df[['r', 'g', 'b']].values

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd], window_name="Lidar Point Cloud")


def plot_lidar_timestamps(df):
    # Plot the number of points over time using the 'epoch' column
    plt.figure(figsize=(10, 6))
    df['epoch'].value_counts().sort_index().plot(kind='bar')
    plt.xlabel('Timestamp (Epoch)')
    plt.ylabel('Number of Points')
    plt.title('Number of LiDAR Points over Time')
    plt.show()


if __name__ == "__main__":
    # Specify the path to your CSV file
    file_path = 'C:\\Users\\localadmin\\PycharmProjects\\Argus\\lidar_data\\Lidar_data_2024-09-26_161608.csv'

    # Read the LiDAR data from the CSV
    df = read_lidar_data(file_path)

    # Visualize the data using Open3D
    visualize_lidar_data(df)

    # Plot the number of points over time (epoch)
    plot_lidar_timestamps(df)
