import time
import open3d as o3d
import numpy as np
from matplotlib import colormaps
from lidar import lidar_setup, lidar_callback, update_lidar_rotation_frequency, lidar_map
from carla_setup import carla_setup
from varjo import varjo_yaw_data
import keyboard
from multiprocessing import Process, Manager

def main():
    global vis, pcd, central_yaw

    # Start Varjo process
    manager = Manager()
    shared_dict = manager.dict()
    varjo_process = Process(target=varjo_yaw_data, args=(shared_dict,))
    varjo_process.start()

    # Get Carla connection
    client, world, blueprint_library, vehicle_list, vehicle1 = carla_setup()

    # Use updated colormap access
    viridis = np.array(colormaps['plasma'].colors)
    vid_range = np.linspace(0.0, 1.0, viridis.shape[0])
    cool_range = np.linspace(0.0, 1.0, viridis.shape[0])
    cool = np.array(colormaps['winter'](cool_range))
    cool = cool[:, :3]

    # Set lidar
    points = 500000
    lidar = lidar_setup(world, blueprint_library, vehicle1, points)
    point_list = o3d.geometry.PointCloud()
    yaw_angle = shared_dict.get('yaw', None)
    central_yaw = -np.radians(yaw_angle) # Central vision yaw angle

    lidar.listen(lambda data: lidar_callback(vid_range, viridis, data, point_list, shared_dict, points, lidar, world, blueprint_library, vehicle1))

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name='Lidar',
        width=960,
        height=540,
        left=480,
        top=270
    )
    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1
    vis.get_render_option().show_coordinate_frame = True
    lidar_map(vis)

    # Initialize gaze lines
    gaze_lines = o3d.geometry.LineSet()
    vis.add_geometry(gaze_lines)
    line_length = 20

    frame = 0
    gaze_refresh_rate = 0.5  # Refresh gaze data every 0.5 seconds
    last_gaze_update_time = time.time()
    while True:
        if keyboard.is_pressed('q'):
            print("Stopping the loop and destroying the lidar sensor...")
            lidar.destroy()
            break
        current_time = time.time()
        if frame == 2:
            vis.add_geometry(point_list)

            # Get the latest yaw angle from the shared dictionary
            # get varjo rotation
        yaw_angle = shared_dict.get('yaw', None)
        yaw_rad = -np.radians(yaw_angle)

        gaze_angle_rad = np.radians(35) # central vision FOV
        gaze_points = np.array([
            [0.0, 0.0, 0.0],
            [line_length * -np.cos(yaw_rad + gaze_angle_rad), line_length * -np.sin(yaw_rad + gaze_angle_rad), 0.0],
            [line_length * -np.cos(yaw_rad - gaze_angle_rad), line_length * -np.sin(yaw_rad - gaze_angle_rad), 0.0]
        ])
        gaze_lines.points = o3d.utility.Vector3dVector(gaze_points)
        gaze_lines.lines = o3d.utility.Vector2iVector(np.array([
            [0, 1],
            [0, 2]
        ]))
        gaze_lines.colors = o3d.utility.Vector3dVector(np.array([
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ]))
        vis.update_geometry(gaze_lines)

        vis.update_geometry(point_list)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.005)
        frame += 1


if __name__ == '__main__':
    main()
