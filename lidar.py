import carla
import numpy as np
import open3d as o3d
from datetime import datetime
import pandas as pd
from preprocess_lidar import filter_ground_points, downsample_point_cloud, segment_clusters, remove_small_clusters, compute_bounding_boxes, match_bounding_boxes, compute_iou

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


def lidar_callback(vid_range, viridis, data, point_list, shared_dict, lidar_live_dict, drivers_gaze=False, lidar_processing=False):
    global prev_bounding_boxes
    downsampling_factor = 6
    # Copy and reshape the LiDAR data
    data = np.copy(np.frombuffer(data.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    if data.size == 0:
        return

    lidar_points = data[:, :-1]
    lidar_points[:, :1] = -lidar_points[:, :1]  # Flip the x axis
    intensity = data[:, -1]
    central_yaw_deg = -shared_dict.get('yaw', None)
    adjusted_intensity = adjust_intensity(lidar_points, intensity, central_yaw_deg)

    intensity_col = 1.0 - np.log(np.clip(adjusted_intensity, a_min=1e-6, a_max=None)) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, vid_range, viridis[:, 0]),
        np.interp(intensity_col, vid_range, viridis[:, 1]),
        np.interp(intensity_col, vid_range, viridis[:, 2])
    ]
    lidar_color = int_color

    if drivers_gaze:
        # Vectorized mask for points in central vision
        point_yaws = np.degrees(np.arctan2(lidar_points[:, 1], lidar_points[:, 0]))
        in_central_vision = np.abs(point_yaws - central_yaw_deg) <= 35

        # Downsample points in central vision
        central_vision_points = lidar_points[in_central_vision]
        central_vision_colors = int_color[in_central_vision]

        downsampled_indices = np.arange(0, len(central_vision_points), downsampling_factor)
        downsampled_points = central_vision_points[downsampled_indices]
        downsampled_colors = central_vision_colors[downsampled_indices]

        # Combine downsampled central vision points with non-downsampled peripheral points
        lidar_points = np.vstack((downsampled_points, lidar_points[~in_central_vision]))
        lidar_color = np.vstack((downsampled_colors, int_color[~in_central_vision]))

    if lidar_processing:
        # Testing recognition of vehicles with Lidar
        # Filter ground points
        filtered_points = filter_ground_points(lidar_points)

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
    point_list.points = o3d.utility.Vector3dVector(lidar_points)
    point_list.colors = o3d.utility.Vector3dVector(lidar_color)

    # Accumulate downsampled points and colors in lidar_live_dict
    # lidar_live_dict['points'].append(combined_points)
    lidar_live_dict['points'].append(lidar_points)
    lidar_live_dict['color'].append(lidar_color)
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


def lidar_callback_wrapped(vid_range, viridis, data, point_list, shared_dict, data_queue, lidar_live_dict, drivers_gaze=False, lidar_processing=False):
    lidar_callback(vid_range, viridis, data, point_list, shared_dict, lidar_live_dict, drivers_gaze=drivers_gaze, lidar_processing=lidar_processing)
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


def adjust_intensity(points, intensity, gaze_angle, field_of_view=60, power_reduction_factor=0.5):
    """
    Adjusts the intensity of the LiDAR points based on the driver's gaze angle and field of view.

    Args:
    - points (np.array): LiDAR points array (N x 3).
    - intensity (np.array): LiDAR intensity array (N,).
    - gaze_angle (float): The driver's gaze angle in degrees.
    - field_of_view (float): The field of view of the driver in degrees.
    - power_reduction_factor (float): Factor by which power is reduced in the driver's field of view.

    Returns:
    - adjusted_intensity (np.array): The adjusted intensity values.
    """
    # Convert the field of view to radians
    field_of_view_rad = np.radians(field_of_view)

    # Compute the angle of each point relative to the driver's gaze
    point_yaws = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    angle_from_gaze = np.abs(point_yaws - gaze_angle)

    # Determine if a point falls within the driver's field of view
    in_fov = angle_from_gaze <= (field_of_view / 2)

    # Adjust the intensity based on whether the point is in the driver's field of view
    adjusted_intensity = np.copy(intensity)
    adjusted_intensity[in_fov] *= power_reduction_factor
    adjusted_intensity[~in_fov] *= (1 / power_reduction_factor)

    return adjusted_intensity