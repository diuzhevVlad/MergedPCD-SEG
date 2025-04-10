import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

# Load and preprocess data
lidar_dir = "../Huawei/dataset/sequences/01/velodyne"
points = np.fromfile(f"{lidar_dir}/000204.bin", dtype=np.float32).reshape(-1, 4)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])

# Ground extraction (adjust Z thresholds)
lower_mask = (points[:, 2] < -1) & (points[:, 2] > -2) & (points[:, 0] > 0)
lowered_pcd = pcd.select_by_index(np.where(lower_mask)[0])
lowered_points = np.asarray(lowered_pcd.points)

# Estimate normals (for slope consistency)
lowered_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
normals = np.asarray(lowered_pcd.normals)

# Find seed point (lowest point)
seed_idx = np.argmin(lowered_points[:, 2])
seed_point = lowered_points[seed_idx]
seed_normal = normals[seed_idx]

# KDTree for neighbor search
tree = KDTree(lowered_points[:, :2])  # Only X,Y for neighbor search

# Parameters
max_z_diff = 0.1          # Max allowed height difference (strict)
max_slope_diff = 0.3      # Max angle difference (radians) between normals
search_radius = 0.5       # Neighbor search radius (adjust based on density)

# Road classification
road_indices = set([seed_idx])
to_visit = [seed_idx]

while to_visit:
    current_idx = to_visit.pop()
    current_point = lowered_points[current_idx]
    current_normal = normals[current_idx]

    # Find neighbors within radius
    neighbor_idxs = tree.query_ball_point(current_point[:2], search_radius)

    for neighbor_idx in neighbor_idxs:
        if neighbor_idx in road_indices:
            continue

        neighbor_point = lowered_points[neighbor_idx]
        neighbor_normal = normals[neighbor_idx]

        # Height difference check
        height_diff = abs(neighbor_point[2] - current_point[2])

        # Slope consistency check (dot product of normals)
        slope_diff = np.arccos(np.clip(np.dot(current_normal, neighbor_normal), -1, 1))

        # Dynamic height threshold (stricter for closer points)
        xy_dist = np.linalg.norm(neighbor_point[:2] - current_point[:2])
        dynamic_z_threshold = max_z_diff * (1 + xy_dist)  # Allow slightly more height diff for distant points

        # Check if point belongs to the road
        if (height_diff < dynamic_z_threshold) and (slope_diff < max_slope_diff):
            road_indices.add(neighbor_idx)
            to_visit.append(neighbor_idx)

# Extract road and pavement points
road_points = lowered_points[list(road_indices)]
pavement_mask = np.ones(len(lowered_points), dtype=bool)
pavement_mask[list(road_indices)] = False
pavement_points = lowered_points[pavement_mask]

# Visualization
road_pcd = o3d.geometry.PointCloud()
road_pcd.points = o3d.utility.Vector3dVector(road_points)
road_pcd.paint_uniform_color([0, 0, 1])  # Blue for road

pavement_pcd = o3d.geometry.PointCloud()
pavement_pcd.points = o3d.utility.Vector3dVector(pavement_points)
pavement_pcd.paint_uniform_color([0, 1, 0])  # Green for pavement

o3d.visualization.draw_geometries(
    [road_pcd, pavement_pcd],
    zoom=0.05,
    front=[-0.1, -0.0, 0.1],
    lookat=[2.1813, 2.0619, 2.0999],
    up=[0.0, -0.0, 1.0],
)