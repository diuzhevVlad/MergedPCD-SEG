from road_det_filt_adv import initialize_model, convert_pc, grid_sample, fit_surface_ransac
import os
import CSF
import numpy as np
from scipy.spatial import KDTree
from sklearn.preprocessing import PolynomialFeatures
import torch


segmodel, seg_head = initialize_model()

def procceed(pcd_file, save_path, id):
    # Load and filter point cloud
    points = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 4)
    points[:,:3] = convert_pc(points[:,:3], [-12.85, 7.85], [-60, 60], points.shape[0] // 480)
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
    ground_pts = points[ground_idx]

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
    
    if road_points.shape[0] > 50:
        # Fit surface and find anomalies
        residuals, ransac_model = fit_surface_ransac(road_points[:, :3], order=2)
        threshold = np.percentile(residuals, 95)

        # Recompute residuals for all ground points
        poly = PolynomialFeatures(degree=2, include_bias=True)
        X_poly = poly.fit_transform(ground_pts[:, :2])
        Y_poly = ransac_model.predict(X_poly)
        residuals = np.abs(Y_poly - ground_pts[:, 2])

        # Prepare data for visualization
        road_points = ground_pts[residuals <= threshold]
        
    tree = KDTree(road_points[:, :2])  # Only use (X, Y) for search
    height_diffs = []

    for i, pt in zip(range(len(road_points)), road_points):
        # Find neighbors within 0.5m radius
        idx = tree.query_ball_point(pt[:2], 0.2)
        neighbors = road_points[idx]

        # Compute height difference with neighbors
        height_diff = np.max(neighbors[:,2]) - np.min(neighbors[:,2])
        height_diffs.append(height_diff)


    height_diffs = np.array(height_diffs)
    np.save(os.path.join(save_path, f"labels/{str(id).zfill(6)}.npy"), height_diffs)
    np.save(os.path.join(save_path, f"feat/{str(id).zfill(6)}.npy"), road_points)


import glob
import tqdm
bag_folders = glob.glob("../Huawei/rosbag2*")

for fold in bag_folders:
    pcd_files = sorted(os.listdir(os.path.join(fold, "pcd")))
    id = 0
    for pcd_file_id in tqdm.tqdm(range(0, len(pcd_files), 10)):
        save_path = f"../IrregData/{os.path.basename(fold)}"
        os.makedirs(os.path.join(save_path, "labels"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "feat"), exist_ok=True)
        procceed(os.path.join(fold, "pcd", pcd_files[pcd_file_id]), save_path, id)
        id+=1

