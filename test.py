import time
import carla
import open3d as o3d
import numpy as np
from matplotlib import colormaps
from lidar import lidar_setup, lidar_callback_wrapped, lidar_map, save_lidar_data
from carla_setup import carla_setup
from varjo import varjo_yaw_data
import keyboard
from multiprocessing import Process, Manager, Event
import queue


def main():
    global vis, pcd, central_yaw, prev_bounding_boxes, prev_position

    # Create shared dictionary to save data between multiprocess
    manager = Manager()
    shared_dict = manager.dict()
    lidar_live_dict = {'epoch': [], 'points': [], 'color': []}

    # Start Varjo process and create event to stop varjo
    stop_event = Event()
    varjo_process = Process(target=varjo_yaw_data, args=(shared_dict, stop_event))
    varjo_process.start()

    # Get Carla connection
    client, world, current_weather, blueprint_library, vehicle_list, vehicle1 = carla_setup()
    fog_density = current_weather.fog_density

    # Use updated colormap access
    viridis = np.array(colormaps['plasma'].colors)
    vid_range = np.linspace(0.0, 1.0, viridis.shape[0])

    # Set lidar
    points = 500000
    frequency = 60
    lidar = lidar_setup(world, blueprint_library, vehicle1, points, frequency, fog_density)
    point_list = o3d.geometry.PointCloud()
    yaw_angle = shared_dict.get('yaw', None)
    central_yaw = -np.radians(yaw_angle) # Central vision yaw angle

    # Create a queue to store LiDAR data
    data_queue = queue.Queue()

    # Wrap the LiDAR callback to use the queue
    lidar.listen(lambda data: lidar_callback_wrapped(vid_range, viridis, data, point_list, shared_dict, data_queue, lidar_live_dict, vehicle1, power_control=True, drivers_gaze=True))

    # Initialize visualizer
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
    while True:
        if keyboard.is_pressed('q'):
            print("Stopping the loop and destroying the lidar sensor...")
            lidar.destroy()
            stop_event.set()  # Signal Varjo process to stop
            varjo_process.join()  # Wait for Varjo process to finish
            # save_lidar_data(lidar_live_dict)  # Save LiDAR data to CSV
            break

        # Update the point cloud if new data is available
        while not data_queue.empty():
            new_data = data_queue.get()
            point_list.points = new_data.points
            point_list.colors = new_data.colors

        if frame == 2:
            vis.add_geometry(point_list)

        yaw_angle = -shared_dict.get('yaw', 0)
        yaw_rad = np.radians(yaw_angle)
        gaze_angle_rad = np.radians(30)  # central vision FOV
        gaze_points = np.array([
            [0.0, 0.0, 0.0],
            [line_length * np.cos(yaw_rad + gaze_angle_rad), line_length * np.sin(yaw_rad + gaze_angle_rad), 0.0],
            [line_length * np.cos(yaw_rad - gaze_angle_rad), line_length * np.sin(yaw_rad - gaze_angle_rad), 0.0]
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

    # Cleanup
    vis.destroy_window()
    varjo_process.terminate()

if __name__ == '__main__':
    main()
