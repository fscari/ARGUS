import carla
import numpy as np
import open3d as o3d
from datetime import datetime
import pandas as pd
from preprocess_lidar import filter_ground_points, downsample_point_cloud, segment_clusters, compute_bounding_boxes, detect_moving_objects, filter_road_points,\
    mark_road_cells, detect_vehicles_in_road_area, create_grid
import globals



prev_position = carla.Location()
# prev_bounding_boxes = []
def lidar_setup(world, blueprint_library, vehicle, points, frequency, fog_density):
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('rotation_frequency', str(frequency))
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', str(points))
    attenuation_rate = 0.004 + (fog_density / 100.0) * 0.04  # scaling factor
    lidar_bp.set_attribute('atmosphere_attenuation_rate', str(attenuation_rate))

    lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=2.5), carla.Rotation(pitch=0, yaw=180, roll=0))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    return lidar


def lidar_callback(vid_range, viridis, data, point_list, shared_dict, lidar_live_dict, vehicle, grid_cache, power_control=False, drivers_gaze=False,
                   bounding_box=False, lidar_processing=False):
    global prev_position # prev_bounding_boxes

    downsampling_factor = 6
    # Copy and reshape the LiDAR data
    data = np.copy(np.frombuffer(data.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    if data.size == 0:
        return

    lidar_points = data[:, :-1]
    lidar_points[:, :1] = -lidar_points[:, :1]  # Flip the x axis
    intensity = data[:, -1]
    central_yaw_deg = -shared_dict.get('yaw', 0)
    threshold = 0.1
    if power_control:
        lidar_points, intensity = adjust_intensity(lidar_points, intensity, central_yaw_deg, threshold)
    else:
        # Filter out points and intensities that don't meet the threshold
        valid_indices = intensity > threshold
        lidar_points = lidar_points[valid_indices]
        intensity = intensity[valid_indices]

    intensity_col = 1.0 - np.log(np.clip(intensity, a_min=1e-6, a_max=None)) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, vid_range, viridis[:, 0]),
        np.interp(intensity_col, vid_range, viridis[:, 1]),
        np.interp(intensity_col, vid_range, viridis[:, 2])
    ]
    lidar_color = int_color

    prev_position = vehicle.get_location()  # Get initial position

    # After some time, e.g., in the next frame
    curr_position = vehicle.get_location()  # Get current position
    velocity = vehicle.get_velocity()  # Get current velocity
    yaw = vehicle.get_transform().rotation.yaw  # Get yaw angle

    # Calculate displacement
    displacement = np.sqrt((curr_position.x - prev_position.x) ** 2 +
                           (curr_position.y - prev_position.y) ** 2 +
                           (curr_position.z - prev_position.z) ** 2)

    # Update previous position for the next iteration
    prev_position = curr_position

    if drivers_gaze:
        # Vectorized mask for points in central vision
        point_yaws = np.degrees(np.arctan2(lidar_points[:, 1], lidar_points[:, 0]))
        in_central_vision = np.abs(point_yaws - central_yaw_deg) <= 30

        # Downsample points in central vision
        central_vision_points = lidar_points[in_central_vision]
        central_vision_colors = int_color[in_central_vision]

        downsampled_indices = np.arange(0, len(central_vision_points), downsampling_factor)
        downsampled_points = central_vision_points[downsampled_indices]
        downsampled_colors = central_vision_colors[downsampled_indices]

        # Combine downsampled central vision points with non-downsampled peripheral points
        lidar_points = np.vstack((downsampled_points, lidar_points[~in_central_vision]))
        lidar_color = np.vstack((downsampled_colors, int_color[~in_central_vision]))

    if bounding_box:
        # Testing recognition of vehicles with Lidar
        # Get vehicle's yaw and convert it to radians
        vehicle_yaw = np.radians(vehicle.get_transform().rotation.yaw)

        # Calculate the angles of each point relative to the vehicle's forward direction
        point_angles = np.arctan2(lidar_points[:, 1], lidar_points[:, 0])

        # Normalize the angles relative to the vehicle's yaw
        relative_angles = point_angles - vehicle_yaw

        # Adjust for LiDAR's 180-degree backward orientation
        relative_angles = relative_angles - np.pi/2

        # Normalize angles to the range [-π, π]
        relative_angles = (relative_angles + np.pi) % (2 * np.pi) - np.pi

        # Create a mask for points within ±90 degrees of the vehicle's forward direction
        roi_mask = np.abs(relative_angles) <= np.pi / 2

        # Filter points within the ROI
        lidar_points_roi = lidar_points[roi_mask]
        lidar_color = lidar_color[roi_mask]

        # Filter ground points
        filtered_points, lidar_color = filter_ground_points(lidar_points_roi, lidar_color)

        # Downsample the entire point cloud
        lidar_points, lidar_color = downsample_point_cloud(filtered_points, lidar_color)
        # Segment and cluster points
        labels = segment_clusters(lidar_points) # dlidar_points

        # Remove noise and small clusters
        # lidar_points, lidar_color = remove_small_clusters(lidar_points, labels, lidar_color)
        # labels = segment_clusters(lidar_points)

        # Compute bounding boxes
        bounding_boxes = compute_bounding_boxes(lidar_points, labels)

        # Check moving bounding boxes
        moving_bounding_boxes = detect_moving_objects(globals.prev_bounding_boxes, bounding_boxes, displacement)

        # Decrease intensity of points within moving objects
        for moving_object in moving_bounding_boxes:
            min_bound = moving_object.min_bound
            max_bound = moving_object.max_bound
            in_moving_object = np.all(np.logical_and(lidar_points >= min_bound, lidar_points <= max_bound), axis=1)
            lidar_points = lidar_points[in_moving_object]
            # lidar_color[in_moving_object] *= 10  # Reduce intensity by 50%
            # print('yes')
            # lidar_color[in_moving_object] = 0  # Reduce intensity by 50%
        globals.prev_bounding_boxes = bounding_boxes
        # print(prev_bounding_boxes)

    if lidar_processing:
        vehicle_yaw = np.radians(vehicle.get_transform().rotation.yaw)
        point_angles = np.arctan2(lidar_points[:, 1], lidar_points[:, 0])
        relative_angles = point_angles - vehicle_yaw
        relative_angles = relative_angles - np.pi / 2
        relative_angles = (relative_angles + np.pi) % (2 * np.pi) - np.pi
        roi_mask = np.abs(relative_angles) <= np.pi / 2
        lidar_points_roi = lidar_points[roi_mask]
        lidar_color_roi = lidar_color[roi_mask]
        road_points, road_colors, non_road_points, non_road_colors = filter_road_points(lidar_points_roi, lidar_color_roi)
        y_min = road_points[:, 1].min()
        y_max = road_points[:, 1].max()

        if grid_cache.needs_update(y_min, y_max) or grid_cache.grid is None:  #
            # Recompute the grid and road cells if needed
            print('creating the grid')
            grid = create_grid(road_points, grid_size=1.0)
            road_cells = mark_road_cells(grid, road_points)

            # Update the cache with the new grid, road cells, and y-values
            grid_cache.update_cache(grid, road_cells)

        vehicle_points = detect_vehicles_in_road_area(non_road_points, grid_cache.road_mask, grid_size=1.0)
        if len(vehicle_points) > 0 and (vehicle_points[:, 1] < -10).any(): #and globals.angle_degrees is None:
            filtered_points = vehicle_points[vehicle_points[:, 1] < -10]
            highest_point = filtered_points[np.argmax(filtered_points[:, 1])]
            angle_radians = np.arctan2(highest_point[1], highest_point[0])
            globals.angle_degrees = np.degrees(angle_radians)
            if globals.time_lidar is None:
                globals.time_lidar = datetime.now()
        # print(globals.angle_degrees)

    # Update the point cloud for visualization
    # point_list.points = o3d.utility.Vector3dVector(combined_points)
    point_list.points = o3d.utility.Vector3dVector(lidar_points) # lidar_points  lidar_points_roi downsampled_points filtered_points  non_road_points
    point_list.colors = o3d.utility.Vector3dVector(lidar_color) # lidar_color road_colors non_road_colors

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
        [1, 0.0, 0.0],
        [0.0, 1, 0.0],
        [0.0, 0.0, 1]
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


def lidar_callback_wrapped(vid_range, viridis, data, point_list, shared_dict, data_queue, lidar_live_dict, vehicle, grid_cache, power_control=False, drivers_gaze=False,
                           bounding_box=False, lidar_processing=False):
    lidar_callback(vid_range, viridis, data, point_list, shared_dict, lidar_live_dict, vehicle, grid_cache, power_control=power_control, drivers_gaze=drivers_gaze,
                   bounding_box=bounding_box, lidar_processing=lidar_processing)
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


def adjust_intensity(points, intensity, gaze_angle, threshold, field_of_view=60):
    """
    Adjusts the intensity of the LiDAR points based on the driver's gaze angle and field of view.

    Args:
    - points (np.array): LiDAR points array (N x 3).
    - intensity (np.array): LiDAR intensity array (N,).
    - gaze_angle (float): The driver's gaze angle in degrees.
    - field_of_view (float): The field of view of the driver in degrees.
    - power_reduction_factor (float): Factor by which power is reduced in the driver's field of view.
    - power_increase_factor (float) :Factor by which power is increased outside the driver's field of view.

    Returns:
    - adjusted_intensity (np.array): The adjusted intensity values.
    """
    power_reduction_factor = 0.5
    power_increase_factor = 1.1
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
    adjusted_intensity[~in_fov] *= power_increase_factor

    # Filter out points and intensities that don't meet the threshold
    valid_indices = adjusted_intensity > threshold
    filtered_points = points[valid_indices]
    filtered_intensity = adjusted_intensity[valid_indices]
    return filtered_points, filtered_intensity
