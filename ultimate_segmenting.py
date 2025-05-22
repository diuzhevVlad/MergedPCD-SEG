from utils import *
import cv2
import CSF
import numpy as np
from scipy.spatial import KDTree
from sklearn.preprocessing import PolynomialFeatures
import torch
from sklearn.cluster import DBSCAN
import glob
from models import initialize_model_pt, PointNet

CLARIFY_WITH_RANSAC = False
POLY_ORDER = 2
CURB_THRES = 0.03 # Height difference to detect curbs
CURB_OUT_THRES = 0.1
WET_REF_THRES = 20
SNOW_REF_THRES = 80
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
COLORS = np.array([ # 0 - dry, 1 - wet, 2 - snow, 3 - pothole, 4 - hill
    [0., 0., 1.],  # normal
    [1., 0., 1.],  # low ref (magenta)
    [0., 1., 1.],  # high ref (cyan)
    [0., 0., 0.],  # pothole
    [1., 0., 0.],  # hill
])


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
    
    if road_points.shape[0] < 50:
        print("Not enough road points!")
        return
    
    if CLARIFY_WITH_RANSAC:
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

    surface_type = np.zeros(road_points.shape[0], np.int8) # 0 - dry, 1 - wet, 2 - snow, 3 - pothole, 4 - hill
    surface_type[road_points[:,3] < WET_REF_THRES] = 1
    surface_type[road_points[:,3] > SNOW_REF_THRES] = 2

        
    tree = KDTree(road_points[:, :2])  # Only use (X, Y) for search
    curb_idxs = []
    height_diffs = []

    for i, pt in zip(range(len(road_points)), road_points):
        idx = tree.query_ball_point(pt[:2], 0.2)
        neighbors = road_points[idx]

        # Compute height difference with neighbors
        height_diff = np.max(neighbors[:,2]) - np.min(neighbors[:,2])
        height_diffs.append(height_diff)

        if height_diff > CURB_THRES and height_diff < CURB_OUT_THRES:
            curb_idxs.append(i)

    height_diffs = np.array(height_diffs)
    curb_idxs = np.array(curb_idxs)

    # start_time = time.time()
    # feat = Tensor(road_points).cuda()
    # edge_index = radius_graph(feat[:,:3], r=0.2, max_num_neighbors=32)
    # with torch.no_grad():
    #     height_diffs = irreg_model(feat, feat[:,:3], edge_index, torch.zeros(feat.shape[0], dtype=int).cuda()).cpu().numpy().reshape(-1)
    # print(time.time() - start_time)

    feats = road_points[curb_idxs,:3]
    db = DBSCAN(eps=0.2, min_samples=10).fit(feats[:,:2])
    labels = db.labels_                                  # −1 = noise
    unique = [c for c in np.unique(labels) if c != -1] 
    clusters = [np.where(labels == c)[0] for c in unique] 

    for k, idxs in enumerate(clusters):
        cl_pts = feats[idxs]
        cl_med = np.median(cl_pts[:, 2])

        centre_xy = np.median(cl_pts[:, :2], axis=0)
        bg_idx = tree.query_ball_point(centre_xy, r=0.4)

        # --- exclude cluster points from the background ring -------------
        cl_road_idx = np.take(curb_idxs, idxs)          # → list of ints
        bg_idx = [i for i in bg_idx if i not in set(cl_road_idx)]
        if not bg_idx:                                  # empty list  → skip
            continue

        bg_med = np.median(road_points[bg_idx, 2])
        dz = cl_med - bg_med
        if dz > 0:
            surface_type[cl_road_idx] = 4
        else:
            surface_type[cl_road_idx] = 3

    img_proj = cv2.cvtColor(draw_points(img, road_points[:, :3], COLORS[surface_type], EXTRINSIC, PROJ_MAT), cv2.COLOR_RGB2BGR)
    cv2.imshow("exp", img_proj)
    cv2.waitKey(0)


if __name__ == "__main__":
    segmodel, seg_head = initialize_model_pt("best_model.pth")
    irreg_model = PointNet().cuda()
    irreg_model.load_state_dict(torch.load("irreg_model.pth"))
    pcd_files = sorted(glob.glob("/home/vladislav/all_data/pc/*.bin"))
    img_files = sorted(glob.glob("/home/vladislav/all_data/img/*.png"))

    cnt = 0
    for pcd_file, img_file in zip(pcd_files, img_files):
        procceed(pcd_file, img_file)
        cnt+=1