import carla
import numpy as np
import open3d as o3d
from datetime import datetime
import pandas as pd
from preprocess_lidar import filter_ground_points, downsample_point_cloud, segment_clusters, remove_small_clusters, compute_bounding_boxes,\
    add_bounding_boxes_to_point_cloud

def lidar_setup(world, blueprint_library, vehicle, points, frequency, fog_density):
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('rotation_frequency', str(frequency))
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', str(points))
    attenuation_rate = 0.004 + (fog_density / 100.0) * 0.04  # Example scaling factor
    lidar_bp.set_attribute('atmosphere_attenuation_rate', str(attenuation_rate))

    lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=2.5), carla.Rotation(pitch=0, yaw=180, roll=0))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    return lidar


def lidar_callback(vid_range, viridis, data, point_list, shared_dict, lidar_live_dict, downsampling_factor=6):
    # Copy and reshape the LiDAR data
    data = np.copy(np.frombuffer(data.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    if data.size == 0:
        return

    points = data[:, :-1]
    points[:, :1] = -points[:, :1]  # Flip the x axis

    intensity = data[:, -1]
    intensity_col = 1.0 - np.log(np.clip(intensity, a_min=1e-6, a_max=None)) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, vid_range, viridis[:, 0]),
        np.interp(intensity_col, vid_range, viridis[:, 1]),
        np.interp(intensity_col, vid_range, viridis[:, 2])
    ]

    central_yaw_deg = -shared_dict.get('yaw', None)

    # Vectorized mask for points in central vision
    point_yaws = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    in_central_vision = np.abs(point_yaws - central_yaw_deg) <= 35

    # Downsample points in central vision
    central_vision_points = points[in_central_vision]
    central_vision_colors = int_color[in_central_vision]

    downsampled_indices = np.arange(0, len(central_vision_points), downsampling_factor)
    downsampled_points = central_vision_points[downsampled_indices]
    downsampled_colors = central_vision_colors[downsampled_indices]

    # Combine downsampled central vision points with non-downsampled peripheral points
    combined_points = np.vstack((downsampled_points, points[~in_central_vision]))
    combined_colors = np.vstack((downsampled_colors, int_color[~in_central_vision]))

    # Testing recognition of vehicles with Lidar
    # Filter ground points
    filtered_points = filter_ground_points(combined_points)

    # Downsample the entire point cloud
    downsampled_points = downsample_point_cloud(filtered_points)

    # Segment and cluster points
    labels = segment_clusters(downsampled_points)

    # Remove noise and small clusters
    filtered_points = remove_small_clusters(downsampled_points, labels)

    # Compute bounding boxes
    bounding_boxes = compute_bounding_boxes(downsampled_points, labels)

    # Update the point cloud for visualization
    # point_list.points = o3d.utility.Vector3dVector(combined_points)
    point_list.points = o3d.utility.Vector3dVector(filtered_points)
    point_list.colors = o3d.utility.Vector3dVector(combined_colors)

    # Accumulate downsampled points and colors in lidar_live_dict
    # lidar_live_dict['points'].append(combined_points)
    lidar_live_dict['points'].append(filtered_points)
    lidar_live_dict['color'].append(combined_colors)
    time_now = datetime.utcnow()
    epoch_time = int((time_now - datetime(1970, 1, 1)).total_seconds() * 1000000000)
    lidar_live_dict['epoch'].append(epoch_time)


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


def lidar_callback_wrapped(vid_range, viridis, data, point_list, shared_dict, data_queue, lidar_live_dict):
    lidar_callback(vid_range, viridis, data, point_list, shared_dict, lidar_live_dict)
    data_queue.put(point_list)


def save_lidar_data(lidar_live_dict):
    # Check if there is data to save
    if not lidar_live_dict['points']:
        print("No LiDAR data to save.")
        return

    # Concatenate all points and colors
    all_points = np.vstack(lidar_live_dict['points'])
    all_colors = np.vstack(lidar_live_dict['color'])
    # Repeat epochs to match the number of points
    epoch_repeated = np.repeat(lidar_live_dict['epoch'], [len(points) for points in lidar_live_dict['points']])

    # Create a DataFrame
    df = pd.DataFrame(all_points, columns=['x', 'y', 'z'])
    df['r'] = all_colors[:, 0]
    df['g'] = all_colors[:, 1]
    df['b'] = all_colors[:, 2]
    df['epoch'] = epoch_repeated

    # Define the file path
    location = 'C:\\Users\\localadmin\\PycharmProjects\\Argus\\lidar_data\\Lidar_data_{}.csv'.format(
        datetime.now().strftime("%Y-%m-%d_%H%M%S")
    )

    # Save to CSV
    df.to_csv(location, index=False)
    print(f"Lidar data saved to {location}")
