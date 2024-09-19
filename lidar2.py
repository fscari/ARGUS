import carla
import numpy as np
import open3d as o3d
from datetime import datetime
import pandas as pd
from preprocess_lidar import filter_road_points,mark_road_cells, detect_vehicles_in_road_area, create_grid
import globals



prev_position = carla.Location()


def lidar_setup(world, blueprint_library, vehicle, points, frequency, fog_density):
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('rotation_frequency', str(frequency))
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', str(points))
    attenuation_rate = 0.004 + (fog_density / 100.0) * 0.04  # scaling factor
    # lidar_bp.set_attribute('atmosphere_attenuation_rate', str(attenuation_rate))

    lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=2.5), carla.Rotation(pitch=0, yaw=180, roll=0))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    return lidar


def lidar_callback(vid_range, viridis, data, point_list, shared_dict, lidar_live_dict, vehicle, grid_cache, fog_density, power_control=False, drivers_gaze=False,
                   lidar_processing=False):
    global prev_position

    downsampling_factor = 3
    # simple attenuation_rate
    # attenuation_rate = 0.004 + (fog_density / 100.0) * 0.04  # scaling factor

    # complex non linear attenuation_rate
    base_attenuation = 0.004
    max_additional_attenuation = 0.1  # Increased to allow higher attenuation
    attenuation_rate = base_attenuation + (fog_density / 100.0) ** 2 * max_additional_attenuation

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
        adjusted_intensity = adjust_intensity(lidar_points, intensity, central_yaw_deg, threshold)
    else:
        adjusted_intensity = intensity

    lidar_points, attenuated_intensity = apply_attenuation(lidar_points, adjusted_intensity, attenuation_rate)

    intensity_col = 1.0 - np.log(np.clip(attenuated_intensity, a_min=1e-6, a_max=None)) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, vid_range, viridis[:, 0]),
        np.interp(intensity_col, vid_range, viridis[:, 1]),
        np.interp(intensity_col, vid_range, viridis[:, 2])
    ]
    lidar_color = int_color
    prev_position = vehicle.get_location()  # Get initial position
    # After some time, e.g., in the next frame
    curr_position = vehicle.get_location()  # Get current position
    # Update previous position for the next iteration
    prev_position = curr_position

    if drivers_gaze:
        # Vectorized mask for points in central vision
        point_yaws = np.degrees(np.arctan2(lidar_points[:, 1], lidar_points[:, 0]))
        in_central_vision = np.abs(point_yaws - central_yaw_deg) <= 30

        # Downsample points in central vision
        central_vision_points = lidar_points[in_central_vision]
        central_vision_colors = lidar_color[in_central_vision]

        num_points = len(central_vision_points)
        downsampled_indices = np.linspace(0, num_points - 1, int(num_points / downsampling_factor)).astype(int)
        downsampled_indices = np.clip(downsampled_indices, 0, num_points - 1)
        downsampled_points = central_vision_points[downsampled_indices]
        downsampled_colors = central_vision_colors[downsampled_indices]

        # Combine downsampled central vision points with non-downsampled peripheral points
        lidar_points = np.vstack((downsampled_points, lidar_points[~in_central_vision]))
        lidar_color = np.vstack((downsampled_colors, lidar_color[~in_central_vision]))

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
        if road_points.size > 0:
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
            if len(vehicle_points) > 0 and (vehicle_points[:, 1] < -10).any():
                filtered_points = vehicle_points[vehicle_points[:, 1] < -10]
                # if (filtered_points[:, 0] > 9).any():
                highest_point = filtered_points[np.argmax(filtered_points[:, 1])]
                angle_radians = np.arctan2(highest_point[1], highest_point[0])
                if globals.time_lidar is None:
                    print(filtered_points)
                    globals.time_lidar = datetime.now()
                    print(f"Lidar: {globals.time_lidar}")
                globals.angle_degrees = np.degrees(angle_radians)
    # # Update the point cloud for visualization
    # point_list.points = o3d.utility.Vector3dVector(lidar_points) # lidar_points  lidar_points_roi downsampled_points filtered_points  non_road_points
    # point_list.colors = o3d.utility.Vector3dVector(lidar_color) # lidar_color road_colors non_road_colors

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


def lidar_callback_wrapped(vid_range, viridis, data, point_list, shared_dict, data_queue, lidar_live_dict, vehicle, grid_cache, fog_density,
                           power_control=False, drivers_gaze=False, lidar_processing=False):
    lidar_callback(vid_range, viridis, data, point_list, shared_dict, lidar_live_dict, vehicle, grid_cache, fog_density,
                   power_control=power_control, drivers_gaze=drivers_gaze, lidar_processing=lidar_processing)
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
    power_reduction_factor = 0.5
    # power_increase_factor = 1.1
    # power_reduction_factor = 0
    # power_increase_factor = 1.5
    power_increase_factor = 2.75

    # Compute the angle of each point relative to the driver's gaze
    point_yaws = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    angle_from_gaze = np.abs(point_yaws - gaze_angle)

    # Determine if a point falls within the driver's field of view
    in_fov = angle_from_gaze <= (field_of_view / 2)

    # Adjust the intensity based on whether the point is in the driver's field of view
    adjusted_intensity = np.copy(intensity)
    adjusted_intensity[in_fov] *= power_reduction_factor
    adjusted_intensity[~in_fov] *= power_increase_factor

    return adjusted_intensity


def apply_attenuation(points, adjusted_intensity, attenuation_rate, intensity_threshold=0.07):
    # Calculate distances of each point from the LiDAR sensor
    distances = np.linalg.norm(points, axis=1)

    # Apply atmospheric attenuation
    attenuated_intensity = adjusted_intensity * np.exp(-attenuation_rate * distances)

    # Apply intensity threshold
    valid_indices = attenuated_intensity >= intensity_threshold
    attenuated_points = points[valid_indices]
    attenuated_intensity = attenuated_intensity[valid_indices]

    return attenuated_points, attenuated_intensity
