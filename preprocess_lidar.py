import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt


class BoundingBox:
    def __init__(self, min_bound, max_bound):
        self.min_bound = np.array(min_bound)
        self.max_bound = np.array(max_bound)

    def get_corners(self):
        # Get the 8 corners of the bounding box
        min_bound = self.min_bound
        max_bound = self.max_bound
        return np.array([
            [min_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], min_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], min_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], max_bound[2]]
        ])

    def volume(self):
        # Calculate the volume of the bounding box
        return np.prod(self.max_bound - self.min_bound)

    def center(self):
        # Calculate the center of the bounding box
        return (self.min_bound + self.max_bound) / 2

def filter_ground_points(points, lidar_color, z_threshold=-2.7): #2.6
    # Keep points with z greater than z_threshold
    new_points = points[points[:, 2] > z_threshold]
    new_colors = lidar_color[points[:, 2] > z_threshold]
    return new_points, new_colors


def downsample_point_cloud(points, lidar_color, voxel_size=0.5):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(lidar_color)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    downsampled_points = np.asarray(downsampled_pcd.points)
    downsampled_colors = np.asarray(downsampled_pcd.colors)

    # print(f"Points before downsampling: {len(points)}, after downsampling: {len(downsampled_points)}")
    return downsampled_points, downsampled_colors


def segment_clusters(points, eps=1, min_samples=5):
    points = np.asarray(points)
    # Perform clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    # Print the number of clusters found
    # unique_labels = np.unique(labels)
    # print(f"Number of clusters found: {len(unique_labels) - (1 if -1 in unique_labels else 0)}")  # Excluding noise
    return labels


def remove_small_clusters(points, labels, lidar_color, min_cluster_size=5):
    unique_labels = np.unique(labels)
    filtered_points = []
    filtered_colors = []
    for label in unique_labels:
        if label == -1:
            continue  # Ignore noise
        cluster_points = points[labels == label]
        cluster_colors = lidar_color[labels == label]
        if len(cluster_points) >= min_cluster_size:
            filtered_points.append(cluster_points)
            filtered_colors.append(cluster_colors)

    filtered_points = np.vstack(filtered_points) if filtered_points else np.empty((0, 3))
    filtered_colors = np.vstack(filtered_colors) if filtered_colors else np.empty((0, 3))

    return filtered_points, filtered_colors


def compute_bounding_boxes(points, labels):
    unique_labels = np.unique(labels)
    bounding_boxes = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:
            continue  # Ignore noise
        cluster_points = points[labels == label]
        if cluster_points.size == 0:
            continue
        min_bound = cluster_points.min(axis=0)
        max_bound = cluster_points.max(axis=0)

        # Create BoundingBox object
        bounding_box = BoundingBox(min_bound, max_bound)
        bounding_boxes.append(bounding_box)
        center = bounding_box.center()
        # print(f"Bounding Box Center: {center}")
    # print("Bounding box: ", len(bounding_boxes))
    return bounding_boxes


def match_bounding_boxes(prev_boxes, curr_boxes):
    matches = []
    for prev_box in prev_boxes:
        for curr_box in curr_boxes:
            iou = compute_iou(prev_box, curr_box)
            if iou > 0.01:  # Example IoU threshold
                matches.append((prev_box, curr_box))
                curr_center = (curr_box.min_bound + curr_box.max_bound) / 2
                # print('matched')
                # print(curr_center)
    return matches


def compute_iou(box1, box2):
    # Calculate IoU between two BoundingBox objects
    corners1 = box1.get_corners()
    corners2 = box2.get_corners()

    # Compute intersection box
    min_point = np.maximum(np.min(corners1, axis=0), np.min(corners2, axis=0))
    max_point = np.minimum(np.max(corners1, axis=0), np.max(corners2, axis=0))

    inter_dim = np.maximum(0, max_point - min_point)
    intersection_volume = np.prod(inter_dim)

    volume1 = box1.volume()
    volume2 = box2.volume()

    union_volume = volume1 + volume2 - intersection_volume

    iou = intersection_volume / union_volume if union_volume > 0 else 0
    return iou


def detect_moving_objects(prev_bounding_boxes, curr_bounding_boxes, displacement, iou_threshold=0.5):
    moving_objects = []
    for prev_box, curr_box in match_bounding_boxes(prev_bounding_boxes, curr_bounding_boxes):
        # Calculate the displacement of the bounding box centers
        prev_center = (prev_box.min_bound + prev_box.max_bound) / 2
        curr_center = (curr_box.min_bound + curr_box.max_bound) / 2
        bbox_displacement = abs(np.linalg.norm(curr_center - prev_center))

        # Adjust for ego vehicle displacement
        relative_displacement = bbox_displacement - np.linalg.norm(displacement)

        # Check if the object is moving
        if relative_displacement > 1 and curr_center[0] < 15 and np.degrees(np.arctan2(curr_center[1], curr_center[0])) > 0:
            moving_objects.append(curr_box)
            angle_rad = np.arctan2(curr_center[1], curr_center[0])
            angle_deg = np.degrees(angle_rad)
            print(angle_deg)
            print(curr_center)

    return moving_objects

def create_bounding_box_lines(bbox: BoundingBox):
    # Create lines to visualize the bounding box
    corners = bbox.get_corners()
    lines = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # Bottom face
        [4, 5], [5, 7], [7, 6], [6, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines)
    )
    # Color for bounding boxes
    line_set.paint_uniform_color([1.0, 0.0, 0.0])  # Red color
    return line_set

def filter_road_points(points, lidar_color, z_threshold=-2.8):
    """
    Filters out the points that are likely part of the road surface.

    Args:
    - points (np.array): N x 3 array of point cloud coordinates.
    - lidar_color (np.array): N x 3 array of color values corresponding to the points.
    - z_threshold (float): Threshold for z-coordinate to filter road points.

    Returns:
    - road_points (np.array): Points that are part of the road surface.
    - road_colors (np.array): Colors corresponding to the road points.
    - non_road_points (np.array): Points that are not part of the road surface.
    - non_road_colors (np.array): Colors corresponding to the non-road points.
    """

    # Separate road and non-road points based on the z threshold
    road_mask = points[:, 2] < z_threshold
    non_road_mask = points[:, 2] >= z_threshold

    road_points = points[road_mask]
    road_colors = lidar_color[road_mask]

    non_road_points = points[non_road_mask]
    non_road_colors = lidar_color[non_road_mask]

    return road_points, road_colors, non_road_points, non_road_colors

def define_road_area(road_points):
    """
    Defines the road area by creating a 2D convex hull around the road points.

    Args:
    - road_points (np.array): N x 3 array of road point cloud coordinates.

    Returns:
    - road_hull (scipy.spatial.ConvexHull): Convex hull representing the road area in the x-y plane.
    """

    # Project road points onto the x-y plane
    road_xy = road_points[:, :2]

    # Compute the convex hull of the road points in the x-y plane
    road_hull = ConvexHull(road_xy)

    return road_hull


def filter_objects_on_road(non_road_points, road_hull, road_height_threshold=-2.7):
    """
    Filters points that are above the road surface but within the road area, likely representing vehicles.

    Args:
    - non_road_points (np.array): N x 3 array of points that are not part of the road.
    - road_hull (scipy.spatial.ConvexHull): Convex hull representing the road area.
    - road_height_threshold (float): The maximum height of the road surface.

    Returns:
    - vehicle_points (np.array): Points that are likely vehicles on the road.
    """

    # Create a polygon from the convex hull vertices
    road_polygon = Polygon(road_hull.points[road_hull.vertices])

    vehicle_mask = []
    for point in non_road_points:
        point_xy = Point(point[:2])
        is_on_road = road_polygon.contains(point_xy)
        is_above_road = point[2] > road_height_threshold

        # Only keep points that are within the road area and above the road height threshold
        vehicle_mask.append(is_on_road and is_above_road)

    vehicle_points = non_road_points[vehicle_mask]
    print(vehicle_points)

    return vehicle_points


def plot_road_hull(road_points, road_hull):
    """
    Plots the road points and their convex hull.

    Args:
    - road_points (np.array): N x 3 array of road point cloud coordinates.
    - road_hull (scipy.spatial.ConvexHull): Convex hull representing the road area in the x-y plane.
    """

    # Project road points onto the x-y plane
    road_xy = road_points[:, :2]

    # Plot the road points
    plt.figure(figsize=(10, 8))
    plt.plot(road_xy[:, 0], road_xy[:, 1], 'o', label='Road Points')

    # Plot the convex hull
    for simplex in road_hull.simplices:
        plt.plot(road_hull.points[simplex, 0], road_hull.points[simplex, 1], 'k-', label='Convex Hull')

    # Optionally, plot the vertices of the convex hull
    plt.plot(road_hull.points[road_hull.vertices, 0], road_hull.points[road_hull.vertices, 1], 'ro', label='Hull Vertices')

    plt.title('Road Area Defined by Convex Hull')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()
