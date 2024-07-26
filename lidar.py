import carla
import numpy as np
import open3d as o3d


def lidar_setup(world, blueprint_library, vehicle, points):
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('rotation_frequency', '50')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', str(points))

    lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=2.5), carla.Rotation(pitch=0, yaw=0, roll=0))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    return lidar


def is_point_in_central_vision(point, central_yaw, angle_threshold=35):
    point_yaw = np.degrees(np.arctan2(point[1], point[0]))
    return abs(point_yaw - central_yaw) <= angle_threshold


def lidar_callback(vid_range, viridis, data, point_list, shared_dict, npoints, lidar, world, blueprint_library, vehicle):
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

    central_yaw = -shared_dict.get('yaw', None)
    central_yaw = -np.radians(central_yaw)
    # Check if points are within central vision
    latest_point = points[-1]
    in_central_vision = is_point_in_central_vision(latest_point, central_yaw)
    print(in_central_vision)
    # Update LiDAR frequency based on points in central vision
    if in_central_vision and lidar.attributes.get('points_per_second') != '200000':
        print("Central vision detected, increasing frequency.")
        update_lidar_rotation_frequency(vid_range, viridis, data, point_list, shared_dict, 200000, lidar, world, blueprint_library, vehicle)
    elif not in_central_vision and lidar.attributes.get('points_per_second') != '500000':
        print("Central vision not detected, reverting frequency.")
        update_lidar_rotation_frequency(vid_range, viridis, data, point_list, shared_dict, 500000, lidar, world, blueprint_library, vehicle)


def update_lidar_rotation_frequency(vid_range, viridis, data, point_list, shared_dict, npoints, lidar, world, blueprint_library, vehicle):
    lidar.destroy()
    lidar = lidar_setup(world, blueprint_library, vehicle, npoints)
    print('new points: ', npoints)
    lidar.listen(lambda data: lidar_callback(vid_range, viridis, data, point_list, shared_dict, npoints, lidar, world, blueprint_library, vehicle))

    return lidar


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
