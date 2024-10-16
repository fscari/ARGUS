import csv
import time
import open3d as o3d
import numpy as np
from matplotlib import colormaps
from lidar import lidar_setup, lidar_callback_wrapped, lidar_map, save_lidar_data
from carla_setup import carla_setup
from varjo import varjo_yaw_data
import keyboard
from multiprocessing import Process, Manager, Event
import queue
import globals
from grid import GridCache
from datetime import datetime
import os


def main(file_path, power_control=False, drivers_gaze=False, bounding_box=False, lp=True, less_Hz=False):
    global vis, pcd, central_yaw, prev_position # prev_bounding_boxes
    grid_cache = GridCache(y_threshold=0.5)

    # Create shared dictionary to save data between multiprocess
    manager = Manager()
    shared_dict = manager.dict()
    lidar_live_dict = {'epoch': [], 'points': [], 'color': []}

    # Start Varjo process and create event to stop varjo
    stop_event = Event()
    varjo_process = Process(target=varjo_yaw_data, args=(shared_dict, stop_event))
    varjo_process.start()

    # Get Carla connection
    client, world, current_weather, blueprint_library, vehicle_list, vehicle1, vehicle2 = carla_setup()
    fog_density = current_weather.fog_density

    # Use updated colormap access
    viridis = np.array(colormaps['plasma'].colors)
    vid_range = np.linspace(0.0, 1.0, viridis.shape[0])

    # Set lidar
    points = 500000
    if drivers_gaze:
        frequency = 30
    else:
        frequency = 20
        # frequency = 60
    lidar = lidar_setup(world, blueprint_library, vehicle1, points, frequency, fog_density)
    point_list = o3d.geometry.PointCloud()
    yaw_angle = shared_dict.get('yaw', 0)
    central_yaw = -np.radians(yaw_angle)

    # Create a queue to store LiDAR data
    data_queue = queue.Queue()

    # Wrap the LiDAR callback to use the queue
    lidar.listen(lambda data: lidar_callback_wrapped(vid_range, viridis, data, point_list, shared_dict, data_queue, lidar_live_dict, vehicle1, grid_cache,
                                                     power_control=power_control, drivers_gaze=drivers_gaze, bounding_box=bounding_box, lidar_processing=lp))

    # Initialize visualizer
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

    # Initialize gaze lines
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
            # save_lidar_data(lidar_live_dict)  # Save LiDAR data to CSV
            break

        # Update the point cloud if new data is available
        while not data_queue.empty():
            new_data = data_queue.get()
            point_list.points = new_data.points
            point_list.colors = new_data.colors

        # if frame == 2:
            # vis.add_geometry(point_list)

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

        # vis.add_geometry(point_list)
        if vehicle2.get_location().x <= 1.75 and arrived is False:
            arrival_time = datetime.now()
            total_time = arrival_time - globals.time_vehicle
            total_time = total_time.total_seconds()
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
        frame += 1

    # Cleanup
    # vis.destroy_window()
    varjo_process.terminate()
    if velocity > 18:
        velocity = 70
    elif velocity > 12:
        velocity = 50
    elif velocity > 7:
        velocity = 30

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([fog_density, velocity, power_control, drivers_gaze, TTA])


if __name__ == '__main__':
    # Define the directory and file name
    directory = directory = r'C:\Users\localadmin\PycharmProjects\Argus\TTA_data'
    date_today = datetime.now().strftime('%Y-%m-%d')
    file_name = fr'results_{date_today}.csv'
    file_path = os.path.join(directory, file_name)
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Fog Percentage", "Velocity", "Power Control Status", "Frequency Control Status", "TTA"])
    count = 0
    # for i in range(60):  # 60
    #     globals.reset_globals()
    #     time.sleep(1)
    #     if count == 0 or count == 4 or count == 8 or count == 12 or count == 16 or count == 20 or count == 24 or count == 28 or count == 32 or count == 36 \
    #             or count == 40 or count == 44 or count == 48 or count == 52 or count == 56:
    #         power_control = False
    #         drivers_gaze = False
    #         print(f"Condition ln")
    #     elif count == 1 or count == 5 or count == 9 or count == 13 or count == 17 or count == 21 or count == 25 or count == 29 or count == 33 or count == 37 \
    #             or count == 41 or count == 45 or count == 49 or count == 53 or count == 57:
    #         power_control = True
    #         drivers_gaze = False
    #         print(f"Condition lpci")
    #     elif count == 2 or count == 6 or count == 10 or count == 14 or count == 18 or count == 22 or count == 26 or count == 30 or count == 34 or count == 38 \
    #          or count == 42 or count == 46 or count == 50 or count == 54 or count == 58:
    #         power_control = False
    #         drivers_gaze = True
    #         print(f"Condition lpcf")
    #     elif count == 3 or count == 7 or count == 11 or count == 15 or count == 19 or count == 23 or count == 27 or count == 31 or count == 35 or count == 39 \
    #          or count == 43 or count == 47 or count == 51 or count == 55 or count == 59:
    #         power_control = True
    #         drivers_gaze = True
    #         print(f"Condition lpcif")
    #     print(f'Power control intensity active: {power_control}')
        # print(f'Power control frequency active: {drivers_gaze}')
    for i in range(180):
        globals.reset_globals()
        time.sleep(1)
        if count % 2 ==0:
            power_control = False
            drivers_gaze = False
            print(f"Condition ln")
        else:
            power_control = True
            drivers_gaze = False
            print(f"Condition lpci")
        print(f'Power control intensity active: {power_control}')
        count += 1
        main(file_path, power_control=power_control, drivers_gaze=drivers_gaze)
