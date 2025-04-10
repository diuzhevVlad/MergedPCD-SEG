import numpy as np
import open3d as o3d
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.cluster import DBSCAN
import torch
from collections import OrderedDict
from model import PointTransformerV3
import CSF

# --- 1. Load Model and Data ---
segmodel = PointTransformerV3(
    in_channels=4,
    order=["z", "z-trans", "hilbert", "hilbert-trans"],
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32),
    enc_patch_size=(128, 128, 128, 128, 128),
    dec_depths=(2, 2, 2, 2),
    dec_channels=(64, 64, 128, 256),
    dec_num_head=(4, 4, 8, 16),
    dec_patch_size=(128, 128, 128, 128),
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    drop_path=0.3,
    shuffle_orders=True,
    pre_norm=True,
    enable_rpe=False,
    enable_flash=False,
    upcast_attention=False,
    upcast_softmax=False,
    cls_mode=False,
    pdnorm_bn=False,
    pdnorm_ln=False,
    pdnorm_decouple=True,
    pdnorm_adaptive=False,
    pdnorm_affine=True,
    pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
).cuda()  # Your existing model setup
seg_head = torch.nn.Linear(64, 19).cuda()

# Load checkpoint (your existing weights loading code)
checkpoint = torch.load("best_model.pth", map_location=lambda storage, loc: storage.cuda())
weight_backbone = OrderedDict()
weight_seg_head = OrderedDict()
for key, value in checkpoint.items():
    if "backbone" in key:
        weight_backbone[key.replace("module.backbone.", "")] = value
    elif "seg_head" in key:
        weight_seg_head[key.replace("module.seg_head.", "")] = value

segmodel.load_state_dict(weight_backbone, strict=True)
seg_head.load_state_dict(weight_seg_head, strict=True)

# --- 2. Load and Preprocess Point Cloud ---
points = np.fromfile("../Huawei/dataset/sequences/01/velodyne/000204.bin", dtype=np.float32).reshape(-1, 4)
points = points[points[:, 0] > 0]  # Remove invalid points
points[:, 3] *= (255 / points[:, 3].max())  # Normalize intensity

# --- 3. Ground Segmentation (CSF) ---
csf = CSF.CSF()
csf.params.bSloopSmooth = False
csf.params.cloth_resolution = 0.5
csf.params.rigidness = 3
csf.setPointCloud(points[:, :3])
ground_idx, non_ground_idx = CSF.VecInt(), CSF.VecInt()
csf.do_filtering(ground_idx, non_ground_idx)
ground_idx, non_ground_idx = np.array(ground_idx), np.array(non_ground_idx)

# --- 4. Road Segmentation (PointTransformer) ---
def fnv_hash_vec(arr):
    """
    FNV64-1A hash for a 2D array.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr

def grid_sample(data, grid_size):
    scaled_coord = data[:,:3] / np.array(grid_size)
    grid_coord = np.floor(scaled_coord).astype(int)
    min_coord = grid_coord.min(0)
    grid_coord -= min_coord
    scaled_coord -= min_coord
    min_coord = min_coord * np.array(grid_size)
    key = fnv_hash_vec(grid_coord)
    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
    idx_select = (
        np.cumsum(np.insert(count, 0, 0)[0:-1])
        + np.random.randint(0, count.max(), count.size) % count
    )
    idx_unique = idx_sort[idx_select]
    return data[idx_unique], grid_coord[idx_unique], min_coord.reshape([1, 3]), idx_sort, count, inverse, idx_select

feat, grid_coord, _, idx_sort, _, inverse, _ = grid_sample(points[ground_idx], 0.05)
feat = torch.as_tensor(feat).cuda()
grid_coord = torch.as_tensor(grid_coord).cuda()
batch = torch.zeros(feat.shape[0], dtype=int).cuda()

data = {"feat": feat, "coord": feat[:, :3], "grid_coord": grid_coord, "batch": batch}
probs = torch.softmax(seg_head(segmodel(data)["feat"]), dim=1)
labels = torch.argmax(probs, dim=1).cpu().numpy()

# Reconstruct full resolution labels
unsorted_inverse = np.empty_like(inverse)
unsorted_inverse[idx_sort] = inverse
labels = labels[unsorted_inverse]

# Create road point cloud
road_mask = labels == 8  # Assuming label 8 is road
road_points = points[ground_idx][road_mask][:, :3]

# --- 5. Local Polynomial Surface Fitting ---
def fit_local_surfaces(points, patch_size=1.0, order=2, distance_thresh=0.05):
    min_xy = np.min(points[:, :2], axis=0)
    max_xy = np.max(points[:, :2], axis=0)
    x_edges = np.arange(min_xy[0], max_xy[0], patch_size)
    y_edges = np.arange(min_xy[1], max_xy[1], patch_size)
    
    residuals = np.full(len(points), np.nan)
    for i in range(len(x_edges) - 1):
        for j in range(len(y_edges) - 1):
            patch_mask = (points[:, 0] >= x_edges[i]) & (points[:, 0] < x_edges[i+1]) & \
                         (points[:, 1] >= y_edges[j]) & (points[:, 1] < y_edges[j+1])
            patch_points = points[patch_mask]
            
            if len(patch_points) < 10:
                continue
                
            X = patch_points[:, :2]
            y = patch_points[:, 2]
            poly = PolynomialFeatures(degree=order)
            X_poly = poly.fit_transform(X)
            
            ransac = RANSACRegressor(
                estimator=LinearRegression(),
                residual_threshold=distance_thresh,
                max_trials=100
            )
            ransac.fit(X_poly, y)
            y_pred = ransac.predict(X_poly)
            residuals[patch_mask] = np.abs(y - y_pred)
    
    return residuals

residuals = fit_local_surfaces(road_points, patch_size=1.0, order=2)

# --- 6. Dynamic Thresholding ---
valid_residuals = residuals[~np.isnan(residuals)]
mad = np.median(np.abs(valid_residuals - np.median(valid_residuals)))
threshold = np.median(valid_residuals) + 3 * mad  # 3 MAD ~= 99.7% for Gaussian
anomaly_mask = residuals > threshold

# --- 7. Post-Processing ---
anomaly_points = road_points[anomaly_mask & ~np.isnan(residuals)]

# Cluster anomalies to remove noise
if len(anomaly_points) > 0:
    clustering = DBSCAN(eps=0.2, min_samples=5).fit(anomaly_points)
    anomaly_labels = clustering.labels_
    valid_anomalies = anomaly_points[anomaly_labels != -1]  # Remove noise (-1 labels)
else:
    valid_anomalies = np.array([])

# --- 8. Visualization ---
pcd_road = o3d.geometry.PointCloud()
pcd_road.points = o3d.utility.Vector3dVector(road_points)
pcd_road.paint_uniform_color([0.6, 0.6, 0.6])  # Gray = road

pcd_anomaly = o3d.geometry.PointCloud()
if len(valid_anomalies) > 0:
    pcd_anomaly.points = o3d.utility.Vector3dVector(valid_anomalies)
pcd_anomaly.paint_uniform_color([1, 0, 0])  # Red = anomalies

pcd_non_ground = o3d.geometry.PointCloud()
pcd_non_ground.points = o3d.utility.Vector3dVector(points[non_ground_idx][:, :3])
pcd_non_ground.paint_uniform_color([0, 0, 1])  # Blue = non-ground

o3d.visualization.draw_geometries(
    [pcd_road, pcd_anomaly, pcd_non_ground],
    zoom=0.05,
    front=[-0.1, -0.0, 0.1],
    lookat=[2.1813, 2.0619, 2.0999],
    up=[0.0, -0.0, 1.0],
)

# --- Optional: Save Results ---
# np.save("anomalies.npy", valid_anomalies)