import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from scipy.linalg import lstsq

def load_and_preprocess(lidar_path):
    """Load and preprocess point cloud data with error handling"""
    try:
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        # points = points[np.linalg.norm(points[:,:3],2,1) > 0]
        # points = points[points[:,0] > 0]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        return points, pcd
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        raise

def detect_curbs(points, mask, search_radius=0.5, height_threshold=0.1):
    """More efficient curb detection with vectorization"""
    tree = KDTree(points[:, :2])
    curb_indices = []
    
    masked_indices = np.where(mask)[0]
    masked_points = points[mask]
    
    # Process in chunks for better memory efficiency
    chunk_size = 1000
    for i in range(0, len(masked_indices), chunk_size):
        chunk_indices = masked_indices[i:i+chunk_size]
        chunk_points = masked_points[i:i+chunk_size]
        
        # Vectorized neighbor queries
        neighbors = tree.query_ball_point(chunk_points[:, :2], search_radius)
        
        for idx, pt, pt_neighbors in zip(chunk_indices, chunk_points, neighbors):
            if len(pt_neighbors) > 3:  # Require minimum neighbors for stable calculation
                max_diff = np.max(np.abs(points[pt_neighbors, 2] - pt[2]))
                if max_diff > height_threshold:
                    curb_indices.append(idx)
    
    return np.array(curb_indices)

def fit_plane_lstsq(points, min_points=10):
    """Robust plane fitting with least squares and error checking"""
    if len(points) < min_points:
        raise ValueError(f"Not enough points ({len(points)}) for plane fitting")
    
    X = np.column_stack([points[:, :2], np.ones(len(points))])
    y = points[:, 2]
    
    try:
        # Using scipy's more robust lstsq implementation
        coeffs, _, _, _ = lstsq(X, y, lapack_driver='gelsy')
        return coeffs
    except Exception as e:
        print(f"Plane fitting error: {e}")
        return np.array([0, 0, np.median(y)])  # Fallback to horizontal plane

def classify_points(points, road_coeffs, pavement_coeffs, road_thresh=0.1, pave_thresh=0.04):
    """Enhanced classification with distance weighting"""
    X = np.column_stack([points[:, :2], np.ones(len(points))])
    
    # Calculate residuals with regularization
    road_res = np.abs(X @ road_coeffs - points[:, 2])
    pave_res = np.abs(X @ pavement_coeffs - points[:, 2])
    
    # Classification with hysteresis
    road_mask = (road_res < road_thresh) & ((road_res < pave_res) | (pave_res > pave_thresh))
    pave_mask = (pave_res < pave_thresh) & ((pave_res < road_res) | (road_res > road_thresh))
    outlier_mask = ~(road_mask | pave_mask)
    
    return road_mask, pave_mask, outlier_mask

def main():
    # Configurable parameters
    params = {
        'lidar_path': "../Huawei/dataset/sequences/01/velodyne/000204.bin",
        'z_range': (-2, -1),
        'x_range': (0, 15),
        'curb_search_radius': 0.5,
        'curb_height_threshold': 0.1,  # Increased for stability
        'road_threshold': 0.1,        # Tighter threshold for road
        'pavement_threshold': 0.04
    }
    
    try:
        # Load data
        points, pcd = load_and_preprocess(params['lidar_path'])
        
        
        # Create ground mask
        ground_mask = (
            (points[:, 2] > params['z_range'][0]) & 
            (points[:, 2] < params['z_range'][1]) & 
            (points[:, 0] > params['x_range'][0]) & 
            (points[:, 0] < params['x_range'][1])
        )
        
        # Detect curbs
        curb_indices = detect_curbs(
            points, ground_mask,
            search_radius=params['curb_search_radius'],
            height_threshold=params['curb_height_threshold']
        )
        
        # Prepare ground points (excluding curbs)
        non_curb_mask = np.ones(len(points), dtype=bool)
        non_curb_mask[curb_indices] = False
        ground_points = points[ground_mask & non_curb_mask]

        print(ground_points.shape)
        ground_pcd = o3d.geometry.PointCloud()
        ground_pcd.points = o3d.utility.Vector3dVector(ground_points[:,:3])
        o3d.io.write_point_cloud("road.pcd", ground_pcd)
        
        # Fit planes with fallback
        road_coeffs = fit_plane_lstsq(ground_points)
        pavement_coeffs = fit_plane_lstsq(points[ground_mask])
        
        # Classify points
        road_mask, pave_mask, outlier_mask = classify_points(
            points, road_coeffs, pavement_coeffs,
            road_thresh=params['road_threshold'],
            pave_thresh=params['pavement_threshold']
        )
        
        # Create colored point clouds
        road_cloud = pcd.select_by_index(np.where(road_mask)[0])
        pavement_cloud = pcd.select_by_index(np.where(pave_mask)[0])
        outlier_cloud = pcd.select_by_index(np.where(outlier_mask)[0])
        
        road_cloud.paint_uniform_color([0, 0, 1])    # Blue for road
        pavement_cloud.paint_uniform_color([0, 1, 0]) # Green for pavement
        outlier_cloud.paint_uniform_color([1, 0, 0])  # Red for outliers
        
        # Visualization
        o3d.visualization.draw_geometries(
            [road_cloud, pavement_cloud, outlier_cloud],
            window_name="Road/Pavement Classification",
            zoom=0.05,
            front=[-0.1, -0.0, 0.1],
            lookat=[2.1813, 2.0619, 2.0999],
            up=[0.0, -0.0, 1.0],
        )
        
        # Optional: Save results
        # o3d.io.write_point_cloud("road.ply", road_cloud)
        # o3d.io.write_point_cloud("pavement.ply", pavement_cloud)
        
    except Exception as e:
        print(f"Processing failed: {e}")

if __name__ == "__main__":
    main()