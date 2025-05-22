import os
import CSF
import cv2
import numpy as np
import rerun as rr
import torch
import tqdm
from sklearn.preprocessing import PolynomialFeatures
from models import initialize_model_pt
import open3d as o3d
from utils import *

# Constants
LINES = 96 # 96 / 192 
MAIN_DIR = "/home/vladislav/Documents/Huawei/rosbag2_2025_02_28-16_36_19__unparsed"
START = None
STOP = None

LIDAR_DIR = os.path.join(MAIN_DIR, "pcd")
IMG_DIR = os.path.join(MAIN_DIR, "images")
ANOMALY_TYPE = "ref" # ref / norm
POLY_ORDER = 2
TIME_INCREMENT = 0.1
PROJ_MAT = np.array([
    [958.25209249, 0, 617.35434211, 0],
    [0, 957.39654998, 357.38740741, 0],
    [0, 0, 1, 0]
])
EXTRINSIC = np.array([
    [-1.2582e-05, -0.999986, 0.00523595, 0.297609],
    [-0.0124447, -0.00523539, -0.999909, -0.0185331],
    [0.999923, -7.77404e-05, -0.0124444, -0.000275199],
    [0, 0, 0, 1]
])


def process_frame(pcd_file, img_file, segmodel, seg_head):
    """Process a single frame of point cloud and image data."""
    # Load and process image
    img = cv2.cvtColor(cv2.imread(os.path.join(IMG_DIR, img_file)), cv2.COLOR_BGR2RGB)

    # Load and filter point cloud
    points = np.fromfile(os.path.join(LIDAR_DIR, pcd_file), dtype=np.float32).reshape(-1, 4)
    points[:,:3] = convert_pc(points[:,:3], [-12.85, 7.85], [-60, 60], LINES)
    points = points[points[:, 0] > 0]  # Remove points with x <= 0

    # Ground segmentation using CSF
    csf = CSF.CSF()
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = 0.5
    csf.params.rigidness = 3
    csf.setPointCloud(points[:, :3])
    ground, non_ground = CSF.VecInt(), CSF.VecInt()
    csf.do_filtering(ground, non_ground)
    
    ground_idx = np.array(ground)
    non_ground_idx = np.array(non_ground)
    ground_pts = points[ground_idx]
    non_ground_pts = points[non_ground_idx]

    

    # Grid sampling and segmentation
    feat, grid_coord, min_coord, idx_sort, count, inverse, idx_select = grid_sample(
        ground_pts, 0.05
    )
    
    # Prepare data for model
    feat = torch.as_tensor(feat).cuda()
    grid_coord = torch.as_tensor(grid_coord).cuda()
    batch = torch.zeros(feat.shape[0], dtype=int).cuda()
    
    data = {
        "feat": feat,
        "coord": feat[:, :3],
        "grid_coord": grid_coord,
        "batch": batch,
    }
    
    # Run segmentation
    probs = torch.softmax(seg_head(segmodel(data)["feat"]), dim=1)
    labels = torch.argmax(probs, dim=1).cpu().numpy()
    
    # Map labels back to original points
    unsorted_inverse = np.empty_like(inverse)
    unsorted_inverse[idx_sort] = inverse
    labels = labels[unsorted_inverse]

    # Separate road and non-road points
    road_points = ground_pts[labels==8] # pcd_ground.select_by_index(np.where(labels == 8)[0])
    non_road_points = ground_pts[labels!=8]
    non_ground_points = non_ground_pts

    
    if road_points.shape[0] > 50:
        # Fit surface and find anomalies
        residuals, ransac_model = fit_surface_ransac(road_points[:, :3], order=POLY_ORDER)
        threshold = np.percentile(residuals, 95)

        # Recompute residuals for all ground points
        poly = PolynomialFeatures(degree=POLY_ORDER, include_bias=True)
        X_poly = poly.fit_transform(ground_pts[:, :2])
        Y_poly = ransac_model.predict(X_poly)
        residuals = np.abs(Y_poly - ground_pts[:, 2])

        # Prepare data for visualization
        road_points = ground_pts[residuals <= threshold]
        non_road_points = ground_pts[residuals > threshold]
        
    # Anomaly detection
    if ANOMALY_TYPE == "ref":
        anomaly_ref_road_mask_low = (road_points[:,3] < 20)
        anomaly_ref_road_mask_high = (road_points[:,3] > 80)

        anomaly_ref_road_points_low = road_points[anomaly_ref_road_mask_low]
        anomaly_ref_road_points_high = road_points[anomaly_ref_road_mask_high]

        normal_road_points = road_points[~(anomaly_ref_road_mask_low | anomaly_ref_road_mask_high)]

        all_pts = np.concatenate([normal_road_points, anomaly_ref_road_points_low, anomaly_ref_road_points_high, non_road_points, non_ground_points])
        all_clr = np.array([
            [0., 0., 1.],  # Road (blue)
            [1., 0., 0.],  # Non-road ground (red)
            [1., 0., 0.],  # Non-ground (red)
            [1., 0., 1.],  # low ref (magenta)
            [0., 1., 1.],  # high ref (cyan)
        ])[np.concatenate([
            np.zeros(normal_road_points.shape[0], dtype=int),
            np.full(anomaly_ref_road_points_low.shape[0], 3, dtype=int),
            np.full(anomaly_ref_road_points_high.shape[0], 4, dtype=int),
            np.ones(non_road_points.shape[0], dtype=int),
            np.full(non_ground_points.shape[0], 2, dtype=int)
        ])]
    elif ANOMALY_TYPE == "norm":
        road_pcd = o3d.geometry.PointCloud()
        road_pcd.points = o3d.utility.Vector3dVector(road_points[:, :3])
        road_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=10))
        normals = np.asarray(road_pcd.normals)
        vertical = np.median(normals, 0)
        vertical /= np.linalg.norm(vertical, 2)
        print(np.linalg.norm(normals[0], 2))
        normal_deviations = 1 - np.abs(normals @ vertical)
        
        anomaly_norm_road_mask = (normal_deviations > 1)
        anomaly_norm_road_points = road_points[anomaly_norm_road_mask]
        normal_road_points = road_points[~anomaly_norm_road_mask]

        all_pts = np.concatenate([road_points, non_road_points, non_ground_points])
        all_clr = np.array([
            [0., 0., 1.],  # Road (blue)
            [1., 0., 0.],  # Non-road ground (red)
            [1., 0., 0.],  # Non-ground (red)
            [1., 1., 0.],  # normal inconsistency (yellow)
        ])[np.concatenate([
            np.zeros(normal_road_points.shape[0], dtype=int),
            np.full(anomaly_norm_road_points.shape[0], 3, dtype=int),
            np.ones(non_road_points.shape[0], dtype=int),
            np.full(non_ground_points.shape[0], 2, dtype=int)
        ])]
    else:
        raise ValueError("Invalid anomaly type")

    return img, all_pts, all_clr, labels, road_points.shape[0]


def main():
    """Main processing loop."""
    rr.init("rerun_road_segmentation", spawn=True)

    segmodel, seg_head = initialize_model_pt("best_model.pth")

    pcd_files = sorted(os.listdir(LIDAR_DIR))
    img_files = sorted(os.listdir(IMG_DIR))
    t = 0

    start = 0 if START is None else START
    end = len(pcd_files) if STOP is None else STOP

    for pcd_file, img_file in tqdm.tqdm(zip(pcd_files[start:end], img_files[start:end])):
        img, all_pts, all_clr, _, road_pts_num = process_frame(pcd_file, img_file, segmodel, seg_head)
        img_proj = draw_points(img, all_pts[:road_pts_num, :3], all_clr[:road_pts_num], EXTRINSIC, PROJ_MAT)
        
        rr.set_time_seconds("sensor_time", t)
        rr.log("img_orig", rr.Image(img))
        rr.log("img_road_proj", rr.Image(img_proj))
        rr.log(
            "pcd",
            rr.Points3D(
                all_pts[:, :3], 
                colors=all_clr, 
                radii=0.05, 
                labels=all_pts[:, 3].astype(np.bytes_)
            )
        )
        t += TIME_INCREMENT
    rr.save(os.path.basename(MAIN_DIR)+".rrd")
    rr.disconnect()


if __name__ == "__main__":
    main()