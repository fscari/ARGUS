import csv
import sys
import open3d as o3d
import numpy as np
import keyboard
import queue
import globals
from matplotlib import colormaps
from lidar2 import lidar_setup, lidar_callback_wrapped, lidar_map, save_lidar_data
from carla_setup import carla_setup
from varjo import varjo_yaw_data
from multiprocessing import Process, Manager, Event
from grid import GridCache
from datetime import datetime
# from start_ARGUS import start_ARGUS


def ARGUS(tta_file_path, fog_density, count, experiment_nr, lidar_directory, power_control=False, drivers_gaze=False, lp=True, freq=False):
    global vis, pcd, central_yaw, prev_position # prev_bounding_boxes
    grid_cache = GridCache(y_threshold=0.5)

    # Create shared dictionary to save data between multiprocess
    manager = Manager()
    shared_dict = manager.dict()
    lidar_live_dict = {'iteration_nr': [], 'epoch': [], 'points': [], 'color': [], 'driver_angle': []}

    # Start Varjo process and create event to stop varjo
    stop_event = Event()
    varjo_process = Process(target=varjo_yaw_data, args=(shared_dict, stop_event))
    varjo_process.start()

    # Get Carla connection
    client, world, current_weather, blueprint_library, vehicle_list, vehicle1, vehicle2 = carla_setup()
    # fog_density = current_weather.fog_density

    # Use updated colormap access
    viridis = np.array(colormaps['plasma'].colors)
    vid_range = np.linspace(0.0, 1.0, viridis.shape[0])

    # Set lidar parameters
    if drivers_gaze:
        frequency = 27.5
        # frequency = 12.5
        points = 687500  # 500000
    else:
        # frequency = 27.5
        frequency = 20
        points = 500000  # 500000
    print(f'Frequency: {frequency}')
    lidar = lidar_setup(world, blueprint_library, vehicle1, points, frequency, fog_density)
    point_list = o3d.geometry.PointCloud()
    yaw_angle = shared_dict.get('yaw', 0)
    central_yaw = -np.radians(yaw_angle)

    # Create a queue to store LiDAR data
    data_queue = queue.Queue()

    # Wrap the LiDAR callback to use the queue
    lidar.listen(lambda data: lidar_callback_wrapped(vid_range, viridis, data, point_list, shared_dict, data_queue, lidar_live_dict, vehicle1, grid_cache, fog_density,
                                                     count, power_control=power_control, drivers_gaze=drivers_gaze, lidar_processing=lp))

    # # Initialize visualizer
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(
    #     window_name='Lidar',
    #     width=960,
    #     height=540,
    #     left=480,
    #     top=270
    # )
    # vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    # vis.get_render_option().point_size = 1
    # vis.get_render_option().show_coordinate_frame = True
    # lidar_map(vis)
    #
    # # Initialize gaze lines
    # gaze_lines = o3d.geometry.LineSet()
    # vis.add_geometry(gaze_lines)

    line_length = 20
    to_check = False
    first_start = True
    arrived = False
    frame = 0
    while True:
        if keyboard.is_pressed('q') or vehicle1.get_location().x == 0:
            if globals.time_lidar is not None:
                TTA = arrival_time - globals.time_lidar
                TTA = TTA.total_seconds()
            else:
                TTA = None
            print(f'Driver warned {TTA} seconds before TTA')
            print(velocity)
            print("Stopping the loop and destroying the lidar sensor...")
            lidar.destroy()
            stop_event.set()  # Signal Varjo process to stop
            varjo_process.join()  # Wait for Varjo process to finish
            save_lidar_data(lidar_live_dict, experiment_nr, lidar_directory, fog_density)  # Save LiDAR data to CSV
            break

        # Update the point cloud if new data is available
        while not data_queue.empty():
            new_data = data_queue.get()
            point_list.points = new_data.points
            point_list.colors = new_data.colors

        vel_v2 = vehicle2.get_velocity()
        vel_v2 = np.sqrt(vel_v2.x**2 + vel_v2.y**2)
        if vel_v2 > 0 and first_start:
            globals.time_vehicle = datetime.now()
            print("start")
            first_start = False
        yaw_angle = -shared_dict.get('yaw', 0)
        yaw_rad = np.radians(yaw_angle)
        gaze_angle_rad = np.radians(30)  # central vision FOV
        gaze_points = np.array([
            [0.0, 0.0, 0.0],
            [line_length * np.cos(yaw_rad + gaze_angle_rad), line_length * np.sin(yaw_rad + gaze_angle_rad), 0.0],
            [line_length * np.cos(yaw_rad - gaze_angle_rad), line_length * np.sin(yaw_rad - gaze_angle_rad), 0.0]
        ])
        # gaze_lines.points = o3d.utility.Vector3dVector(gaze_points)
        # gaze_lines.lines = o3d.utility.Vector2iVector(np.array([
        #     [0, 1],
        #     [0, 2]
        # ]))
        # gaze_lines.colors = o3d.utility.Vector3dVector(np.array([
        #     [1.0, 1.0, 0.0],
        #     [1.0, 1.0, 0.0]
        # ]))
        # vis.update_geometry(gaze_lines)
        #
        # vis.add_geometry(point_list)
        if vehicle2.get_location().x <= 1.75 and arrived is False:
            arrival_time = datetime.now()
            total_time = arrival_time - globals.time_vehicle
            arrived = True
            velocity = vel_v2
            print(f"arrived: {arrival_time}")

        if globals.angle_degrees is not None and to_check is False:
            print(f"Vehicle approaching from the right with an angle of {abs(np.round(globals.angle_degrees))}!")
            time_diff_lidar = globals.time_lidar - globals.time_vehicle
            time_diff_lidar = time_diff_lidar.total_seconds()
            if abs(yaw_angle) > abs(globals.angle_degrees):
                to_check = True
                globals.time_check = datetime.now()
                time_diff_driver = globals.time_check - globals.time_lidar
                time_diff_general = globals.time_check - globals.time_vehicle
                time_diff_driver = time_diff_driver.total_seconds()
                time_diff_general = time_diff_general.total_seconds()
                print("Vehicle seen")
                print(f"LiDAR needed {time_diff_lidar}s")
                print(f"Driver needed {time_diff_driver}s")
                print(f"Total time needed {time_diff_general}s")

        # vis.poll_events()
        # vis.update_renderer()
        # time.sleep(0.005)
        # frame += 1

        # Cleanup
    # vis.destroy_window()
    varjo_process.terminate()
    if velocity > 18:
        velocity = 70
    elif velocity > 12:
        velocity = 50
    elif velocity > 7:
        velocity = 30

    with open(tta_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([count, fog_density, velocity, power_control, drivers_gaze, TTA])


# if __name__ == "__main__":
#     iteration_nr_joan = sys.argv[1] if len(sys.argv) > 1 else None
#     start_ARGUS(iteration_nr_joan)
