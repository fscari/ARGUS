import carla
import numpy as np
import open3d as o3d


def lidar_setup(world, blueprint_library, vehicle, points, yaw):
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('rotation_frequency', '20')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', str(points))

    lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=2.5), carla.Rotation(pitch=0, yaw=180, roll=0))
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

    central_yaw_deg = -shared_dict.get('yaw', None)

    colors = []
    for i, point in enumerate(points):
        if is_point_in_central_vision(point, central_yaw_deg):
            colors.append([0, 0, 1])  # Blue for points within central vision
        else:
            colors.append([int_color[i, 0], int_color[i, 1], int_color[i, 2]])

    point_list.colors = o3d.utility.Vector3dVector(colors)
    point_list.points = o3d.utility.Vector3dVector(points)
    print(f"Processing {len(data)} points")


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

def lidar_callback_wrapped(vid_range, viridis, data, point_list, shared_dict, npoints, lidar, world, blueprint_library, vehicle, data_queue):
    new_point_list = o3d.geometry.PointCloud()
    lidar_callback(vid_range, viridis, data, new_point_list, shared_dict, npoints, lidar, world, blueprint_library, vehicle)
    data_queue.put(new_point_list)
