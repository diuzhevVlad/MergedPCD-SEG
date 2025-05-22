from utils import *
import open3d as o3d
import cv2
import CSF
import numpy as np
from scipy.spatial import KDTree
from sklearn.preprocessing import PolynomialFeatures
import torch
from sklearn.cluster import DBSCAN


from models import initialize_model_pt, PointNet

import time
POLY_ORDER = 2
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

segmodel, seg_head = initialize_model_pt("best_model.pth")
irreg_model = PointNet().cuda()
irreg_model.load_state_dict(torch.load("irreg_model.pth"))

def procceed(pcd_file, img_file):
    
    img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)

    # Load and filter point cloud
    points = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 4).astype(np.float32)
    if np.isnan(points).sum() > 0:
        points = np.fromfile(pcd_file).reshape(-1, 4).astype(np.float32)

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
        road_res = residuals[residuals <= threshold]
        
    tree = KDTree(road_points[:, :2])  # Only use (X, Y) for search

    curb_threshold = 0.03 # Height difference to detect curbs
    curb_idxs = []

    start_time = time.time()
    height_diffs = []

    for i, pt in zip(range(len(road_points)), road_points):
        idx = tree.query_ball_point(pt[:2], 0.2)
        neighbors = road_points[idx]

        # Compute height difference with neighbors
        height_diff = np.max(neighbors[:,2]) - np.min(neighbors[:,2])
        height_diffs.append(height_diff)

        if height_diff > curb_threshold and height_diff < 0.1:
            curb_idxs.append(i)

    height_diffs = np.array(height_diffs)
    print(time.time() - start_time)

    # start_time = time.time()
    # feat = Tensor(road_points).cuda()
    # edge_index = radius_graph(feat[:,:3], r=0.2, max_num_neighbors=32)
    # with torch.no_grad():
    #     height_diffs = irreg_model(feat, feat[:,:3], edge_index, torch.zeros(feat.shape[0], dtype=int).cuda()).cpu().numpy().reshape(-1)
    # print(time.time() - start_time)

    feats = road_points[curb_idxs,:3]
    curb_res = road_res[curb_idxs]
    db = DBSCAN(eps=0.2, min_samples=10).fit(feats[:,:2])
    labels = db.labels_                                  # −1 = noise
    unique = [c for c in np.unique(labels) if c != -1] 
    clusters = [np.where(labels == c)[0] for c in unique] 

    all_tree = KDTree(road_points[:, :2])
    n_clust = len(clusters)
    cluster_colors  = [np.array([128,128,128])] * n_clust   # default grey
    for k, idxs in enumerate(clusters):
        cl_pts = feats[idxs]
        cl_med = np.median(cl_pts[:, 2])

        centre_xy = np.median(cl_pts[:, :2], axis=0)
        bg_idx = all_tree.query_ball_point(centre_xy, r=0.4)

        # --- exclude cluster points from the background ring -------------
        cl_road_idx = np.take(curb_idxs, idxs)          # → list of ints
        bg_idx = [i for i in bg_idx if i not in set(cl_road_idx)]
        if not bg_idx:                                  # empty list  → skip
            continue

        bg_med = np.median(road_points[bg_idx, 2])
        dz = cl_med - bg_med
        if dz > 0:
            cluster_colors[k] = np.array([255, 0, 0])   # red bump/curb
        else:
            cluster_colors[k] = np.array([0, 0, 255])

    irregs = []
    irregs_clr = []
    for k, idxs in enumerate(clusters):
        irregs.append(feats[idxs].copy())
        irregs_clr.append(np.tile(cluster_colors[k], (len(idxs),1)))
    irregs = np.vstack(irregs)
    irregs_clr = np.vstack(irregs_clr)

    img_proj = cv2.cvtColor(draw_points(img, irregs, irregs_clr, EXTRINSIC, PROJ_MAT), cv2.COLOR_RGB2BGR)
    # img_proj = cv2.cvtColor(draw_points(img, road_points[:, :3], np.array(list(map(value_to_colour, height_diffs)))), cv2.COLOR_RGB2BGR)
    # clrbr = colourbar_png(200)
    # h, w = clrbr.shape[:2]
    # y1, y2 = img_proj.shape[0]-h-10, img_proj.shape[0]-10
    # x1, x2 = img_proj.shape[1]-w-10, img_proj.shape[1]-10
    # img_proj[y1:y2, x1:x2][(clrbr[:,:,:3]!=255).all(axis=2)] = clrbr[:,:,:3][(clrbr[:,:,:3]!=255).all(axis=2)]

    # curb_init_idxs = np.array(curb_idxs)
    # road_mask[curb_init_idxs] = False
    # img_proj = draw_points(img, road_points[road_mask, :3], np.array([[0.,0.,255.]]*np.count_nonzero(road_mask)))
    # img_proj = draw_points(img_proj, road_points[curb_init_idxs, :3], np.array([[255.,0.,0.]]*len(curb_init_idxs)))

    cv2.imshow("exp", img_proj)
    cv2.waitKey(0)


pcd_files = ["../Huawei/special_cases/flat/000000.bin", "../Huawei/special_cases/pothole/000000.bin", "../Huawei/special_cases/unpaved/000000.bin", "../Huawei/special_cases/speedbump/000000.bin"]
img_files = ["../Huawei/special_cases/flat/000000.png", "../Huawei/special_cases/pothole/000000.png", "../Huawei/special_cases/unpaved/000000.png", "../Huawei/special_cases/speedbump/000000.png"]
import glob
pcd_files = sorted(glob.glob("/home/vladislav/all_data/pc/*.bin"))[59:]
img_files = sorted(glob.glob("/home/vladislav/all_data/img/*.png"))[59:]

cnt = 0
for pcd_file, img_file in zip(pcd_files, img_files):
    procceed(pcd_file, img_file)
    cnt+=1
