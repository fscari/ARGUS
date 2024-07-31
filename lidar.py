import carla
import numpy as np
import open3d as o3d
from datetime import datetime
import pandas as pd


def lidar_setup(world, blueprint_library, vehicle, points, frequency):
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('rotation_frequency', str(frequency))
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', str(points))

    lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=2.5), carla.Rotation(pitch=0, yaw=180, roll=0))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    return lidar


def lidar_callback(vid_range, viridis, data, point_list, shared_dict, npoints, lidar, world, blueprint_library, vehicle, lidar_live_dict):
    data = np.copy(np.frombuffer(data.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    if data.size == 0:
        return
    points = data[:, :-1]
    points[:, :1] = -points[:, :1]  # Flip the x axis

    point_list.points = o3d.utility.Vector3dVector(points)

    intensity = data[:, -1]
    intensity_col = 1.0 - np.log(np.clip(intensity, a_min=1e-6, a_max=None)) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, vid_range, viridis[:, 0]),
        np.interp(intensity_col, vid_range, viridis[:, 1]),
        np.interp(intensity_col, vid_range, viridis[:, 2])
    ]
    point_list.colors = o3d.utility.Vector3dVector(int_color)

    central_yaw_deg = -shared_dict.get('yaw', None)

    # Vectorized mask for points in central vision
    point_yaws = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    in_central_vision = np.abs(point_yaws - central_yaw_deg) <= 35
    # Apply colors conditionally
    colors = int_color.copy()  # Start with intensity-based colors
    colors[in_central_vision] = [0, 0, 1]  # Blue for points in central vision
    point_list.colors = o3d.utility.Vector3dVector(colors)
    # point_list.colors = o3d.utility.Vector3dVector(int_color)
    point_list.points = o3d.utility.Vector3dVector(points)

    # Accumulate points and colors in shared_dict
    lidar_live_dict['points'].append(points)
    lidar_live_dict['color'].append(colors)
    time_now = datetime.utcnow()
    epoch_time = int((time_now - datetime(1970, 1, 1)).total_seconds() * 1000000000)
    lidar_live_dict['epoch'].append(epoch_time)
    # print(points)
    print(lidar_live_dict['epoch'])



def lidar_map(vis):
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]
    ]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]))
    vis.add_geometry(axis)


def lidar_callback_wrapped(vid_range, viridis, data, point_list, shared_dict, npoints, lidar, world, blueprint_library, vehicle, data_queue, lidar_live_dict):
    lidar_callback(vid_range, viridis, data, point_list, shared_dict, npoints, lidar, world, blueprint_library, vehicle, lidar_live_dict)
    data_queue.put(point_list)


def save_lidar_data(lidar_live_dict):
    # Check if there is data to save
    if not lidar_live_dict['points']:
        print("No LiDAR data to save.")
        return

    # # Concatenate all points and colors
    # all_points = np.vstack(lidar_live_dict['points'])
    # all_colors = np.vstack(lidar_live_dict['color'])
    # # all_epochs = np.array(lidar_live_dict['epoch'])
    # #
    # # # Create a DataFrame
    # df = pd.DataFrame(all_points, columns=['x', 'y', 'z'])
    # df['r'] = all_colors[:, 0]
    # df['g'] = all_colors[:, 1]
    # df['b'] = all_colors[:, 2]
    # df['epoch'] = lidar_live_dict['epoch']
    df = pd.DataFrame.from_dict(lidar_live_dict)
    # Define the file path
    location = 'C:\\Users\\localadmin\\PycharmProjects\\Argus\\lidar_data\\Lidar_data_{}.csv'.format(
        datetime.now().strftime("%Y-%m-%d_%H%M%S")
    )

    # Save to CSV
    df.to_csv(location, index=False)
    print(f"Lidar data saved to {location}")
