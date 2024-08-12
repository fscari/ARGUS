import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN


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

def filter_ground_points(points, z_threshold=-2.6):
    # Keep points with z greater than z_threshold
    return points[points[:, 2] > z_threshold]


def downsample_point_cloud(points, voxel_size=0.5):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    downsampled_points = np.asarray(downsampled_pcd.points)
    # print(f"Points before downsampling: {len(points)}, after downsampling: {len(downsampled_points)}")
    return downsampled_points


def segment_clusters(points, eps=2, min_samples=5):
    points = np.asarray(points)
    # Perform clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    # Print the number of clusters found
    # unique_labels = np.unique(labels)
    # print(f"Number of clusters found: {len(unique_labels) - (1 if -1 in unique_labels else 0)}")  # Excluding noise
    return labels


def remove_small_clusters(points, labels, min_cluster_size=2):
    unique_labels = np.unique(labels)
    filtered_points = []
    for label in unique_labels:
        if label == -1:
            continue  # Ignore noise
        cluster_points = points[labels == label]
        if len(cluster_points) >= min_cluster_size:
            filtered_points.append(cluster_points)
    return np.vstack(filtered_points) if filtered_points else np.empty((0, 3))


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

    return bounding_boxes


def match_bounding_boxes(prev_boxes, curr_boxes):
    matches = []
    for prev_box in prev_boxes:
        for curr_box in curr_boxes:
            iou = compute_iou(prev_box, curr_box)
            if iou > 0.75:  # Example IoU threshold
                matches.append((prev_box, curr_box))
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
        bbox_displacement = np.linalg.norm(curr_center - prev_center)

        # Adjust for ego vehicle displacement
        relative_displacement = bbox_displacement - np.linalg.norm(displacement)

        # Check if the object is moving
        if relative_displacement > 1.15:
            moving_objects.append(curr_box)
            print(relative_displacement)

    return moving_objects
