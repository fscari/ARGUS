import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import pandas as pd
import numpy as np


def load_lidar_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Data loaded from {file_path}")
    return df


class LidarApp:
    def __init__(self, df):
        self.df = df
        self.current_epoch_idx = 0
        self.epochs = df['epoch'].unique()

        # Create GUI application
        self.app = gui.Application.instance
        self.app.initialize()

        # Create a window
        self.window = gui.Application.instance.create_window("Lidar Data Viewer", 1024, 768)
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.widget3d)

        # Set up point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.widget3d.scene.add_geometry("PointCloud", self.pcd, rendering.MaterialRecord())

        # Set up the slider
        self.slider = gui.Slider(gui.Slider.INT)
        self.slider.set_limits(0, len(self.epochs) - 1)
        self.slider.int_value = 0
        self.slider.set_on_value_changed(self.on_slider_change)

        # Add the slider to the window
        layout = gui.VGrid(2)
        layout.add_child(gui.Label("Timestamp"))
        layout.add_child(self.slider)
        self.window.add_child(layout)

        # Load initial point cloud data
        self.update_point_cloud()

    def on_slider_change(self, value):
        self.current_epoch_idx = value
        self.update_point_cloud()

    def update_point_cloud(self):
        epoch = self.epochs[self.current_epoch_idx]
        points_df = self.df[self.df['epoch'] == epoch]

        points = points_df[['x', 'y', 'z']].values
        colors = points_df[['r', 'g', 'b']].values

        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        self.widget3d.scene.remove_geometry("PointCloud")
        self.widget3d.scene.add_geometry("PointCloud", self.pcd, rendering.MaterialRecord())
        self.widget3d.force_redraw()

    def run(self):
        self.app.run()


if __name__ == "__main__":
    file_path = 'C:\\Users\\localadmin\\PycharmProjects\\Argus\\lidar_data\\Lidar_data_2024-09-26_161608.csv'
    df = load_lidar_data(file_path)
    app = LidarApp(df)
    app.run()
