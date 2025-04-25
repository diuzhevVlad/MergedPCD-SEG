from road_det_filt_adv import initialize_model, convert_pc, grid_sample, fit_surface_ransac, draw_points
import open3d as o3d
import cv2
import CSF
import numpy as np
from scipy.spatial import KDTree
from sklearn.preprocessing import PolynomialFeatures
import torch

segmodel, seg_head = initialize_model()

def procceed(pcd_file, img_file):
    
    img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)

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
        residuals, ransac_model = fit_surface_ransac(road_points[:, :3], order=2)
        threshold = np.percentile(residuals, 95)

        # Recompute residuals for all ground points
        poly = PolynomialFeatures(degree=2, include_bias=True)
        X_poly = poly.fit_transform(ground_pts[:, :2])
        Y_poly = ransac_model.predict(X_poly)
        residuals = np.abs(Y_poly - ground_pts[:, 2])

        # Prepare data for visualization
        road_points = ground_pts[residuals <= threshold]
        non_road_points = ground_pts[residuals > threshold]
        


    road_mask = np.ones(len(road_points),dtype=bool)
    tree = KDTree(road_points[:, :2])  # Only use (X, Y) for search

    curb_threshold = 0.05 # Height difference to detect curbs
    curb_idxs = []

    for i, pt in zip(range(len(road_points)), road_points):
        # Find neighbors within 0.5m radius
        idx = tree.query_ball_point(pt[:2], 0.3)
        neighbors = road_points[idx]

        # Compute height difference with neighbors
        height_diff = np.max(neighbors[:,2]) - np.min(neighbors[:,2])

        if height_diff > curb_threshold:
            curb_idxs.append(i)

    curb_init_idxs = np.array(curb_idxs)
    road_mask[curb_init_idxs] = False
    img_proj = draw_points(img, road_points[road_mask, :3], np.array([[0.,0.,255.]]*np.count_nonzero(road_mask)))
    img_proj = draw_points(img_proj, road_points[curb_init_idxs, :3], np.array([[255.,0.,0.]]*len(curb_init_idxs)))

    cv2.imshow("exp", cv2.cvtColor(img_proj, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


pcd_files = ["../Huawei/special_cases/flat/000000.bin", "../Huawei/special_cases/pothole/000000.bin", "../Huawei/special_cases/unpaved/000000.bin", "../Huawei/special_cases/speedbump/000000.bin"]
img_files = ["../Huawei/special_cases/flat/000000.png", "../Huawei/special_cases/pothole/000000.png", "../Huawei/special_cases/unpaved/000000.png", "../Huawei/special_cases/speedbump/000000.png"]
cnt = 0
for pcd_file, img_file in zip(pcd_files, img_files):
    procceed(pcd_file, img_file)
