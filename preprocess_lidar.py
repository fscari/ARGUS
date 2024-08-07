import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN


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


def segment_clusters(points, eps=10, min_samples=10):
    points = np.asarray(points)
    # Perform clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    # Print the number of clusters found
    # unique_labels = np.unique(labels)
    # print(f"Number of clusters found: {len(unique_labels) - (1 if -1 in unique_labels else 0)}")  # Excluding noise
    return labels


def remove_small_clusters(points, labels, min_cluster_size=100):
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
    print(f"Points shape: {points.shape}")
    print(f"Labels shape: {labels.shape}")

    unique_labels = np.unique(labels)
    print(f"Unique labels: {unique_labels}")
    bounding_boxes = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:
            continue  # Ignore noise
        cluster_points = points[labels == label]
        if cluster_points.size == 0:
            continue
        # min_bound = cluster_points.min(axis=0)
        # max_bound = cluster_points.max(axis=0)
        # bounding_boxes.append((min_bound, max_bound))
        # Create a bounding box for the cluster
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(cluster_points))
        bbox.color = (1, 0, 0)  # Set color to red
        bounding_boxes.append(bbox)
    return bounding_boxes
