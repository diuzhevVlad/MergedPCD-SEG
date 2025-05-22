import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import CSF
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor, LinearRegression
from models import PointTransformerV3
import torch
from collections import OrderedDict

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
).cuda()
seg_head = torch.nn.Linear(64, 19).cuda()

checkpoint = torch.load(
    "best_model.pth", map_location=lambda storage, loc: storage.cuda()
)

weight_backbone = OrderedDict()
weight_seg_head = OrderedDict()

for key, value in checkpoint.items():
    if "backbone" in key:
        weight_backbone[key.replace("module.backbone.", "")] = value
    elif "seg_head" in key:
        weight_seg_head[key.replace("module.seg_head.", "")] = value

load_state_info1 = segmodel.load_state_dict(weight_backbone, strict=True)
load_state_info2 = seg_head.load_state_dict(weight_seg_head, strict=True)
assert load_state_info1 and load_state_info2

lidar_dir = "../Huawei/dataset/sequences/04/velodyne"
ORDER = 2

points = np.fromfile(f"{lidar_dir}/001235.bin", dtype=np.float32).reshape(-1, 4)
points = points[(points[:,0] > 0)]
# points[:,3] *= (255 / points[:,3].max())

csf = CSF.CSF()
csf.params.bSloopSmooth = False
csf.params.cloth_resolution = 0.5
csf.params.rigidness = 3

csf.setPointCloud(points[:,:3])
ground = CSF.VecInt()  # a list to indicate the index of ground points after calculation
non_ground = CSF.VecInt() # a list to indicate the index of non-ground points after calculation
csf.do_filtering(ground, non_ground) # do actual filtering.

ground_idx = np.array(ground)
non_ground_idx = np.array(non_ground)

ground_pts = points[:,:3][ground_idx]
non_ground_pts = points[:,:3][non_ground_idx]
pcd_ground = o3d.geometry.PointCloud()
pcd_non_ground = o3d.geometry.PointCloud()

pcd_ground.points = o3d.utility.Vector3dVector(ground_pts)
pcd_non_ground.points = o3d.utility.Vector3dVector(non_ground_pts)

pcd_ground.paint_uniform_color([0.0, 0, 1.0])
pcd_non_ground.paint_uniform_color([1.0, 0, 0.0])


o3d.visualization.draw_geometries(
    [pcd_ground, pcd_non_ground],
    zoom=0.05,
    front=[-0.1, -0.0, 0.1],
    lookat=[2.1813, 2.0619, 2.0999],
    up=[0.0, -0.0, 1.0],
)

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

feat, grid_coord, min_coord, idx_sort, count, inverse, idx_select = grid_sample(points[ground_idx], 0.05)
# print(heaped_pts.shape, feat.shape)
feat = torch.as_tensor(feat).cuda()
grid_coord = torch.as_tensor(grid_coord).cuda()
batch = torch.zeros(feat.shape[0],dtype=int).cuda()
data = {
    "feat": feat,
    "coord": feat[:,:3],
    "grid_coord": grid_coord,
    "batch": batch,
}
probs = torch.softmax(seg_head(segmodel(data)["feat"]), dim=1)
labels = torch.argmax(probs, dim=1).cpu().numpy()
unsorted_inverse = np.empty_like(inverse)
unsorted_inverse[idx_sort] = inverse
labels = labels[unsorted_inverse]

pcd_road = pcd_ground.select_by_index(np.where(labels==8)[0])
pcd_non_road = pcd_ground.select_by_index(np.where(labels==8)[0], invert=True)

pcd_road.paint_uniform_color([0.0, 1.0, 0.0])
pcd_non_road.paint_uniform_color([0.0, 0, 1.0])

o3d.visualization.draw_geometries(
    [pcd_road, pcd_non_road, pcd_non_ground],
    zoom=0.05,
    front=[-0.1, -0.0, 0.1],
    lookat=[2.1813, 2.0619, 2.0999],
    up=[0.0, -0.0, 1.0],
)

road_points = np.asarray(pcd_road.points)

def fit_surface_ransac(points, order, distance_threshold=0.05):
    """
    Fit a polynomial surface using RANSAC.
    Returns: Residuals (deviation from the fitted surface).
    """
    X = points[:, :2]  # XY coordinates (input features)
    y = points[:, 2]   # Z (height, target)
    
    # Create polynomial features (e.g., [1, x, y, x², xy, y²] for order=2)
    poly = PolynomialFeatures(degree=order, include_bias=True)
    X_poly = poly.fit_transform(X)
    
    # RANSAC regressor (robust to outliers)
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        residual_threshold=distance_threshold,
        max_trials=100
    )
    ransac.fit(X_poly, y)
    
    # Predict and compute residuals
    y_pred = ransac.predict(X_poly)
    residuals = np.abs(y - y_pred)
    
    return residuals, ransac

# Fit surface and compute residuals
residuals, ransac_model = fit_surface_ransac(road_points, order=ORDER)

# Set threshold (e.g., 95th percentile of residuals)
threshold = np.percentile(residuals, 95)
anomaly_mask = residuals > threshold

# Get anomaly points
anomaly_points = road_points[anomaly_mask]

pcd_road_pure = pcd_road.select_by_index(np.where(anomaly_mask)[0], invert=True)
pcd_road_anom = pcd_road.select_by_index(np.where(anomaly_mask)[0])
pcd_road_anom.paint_uniform_color([0.0, 1.0, 1.0])


o3d.visualization.draw_geometries(
    [pcd_road_pure, pcd_road_anom,  pcd_non_road, pcd_non_ground],
    zoom=0.05,
    front=[-0.1, -0.0, 0.1],
    lookat=[2.1813, 2.0619, 2.0999],
    up=[0.0, -0.0, 1.0],
)

poly = PolynomialFeatures(degree=ORDER, include_bias=True)
X_poly = poly.fit_transform(ground_pts[:,:2])
Y_poly = ransac_model.predict(X_poly)
residuals = np.abs(Y_poly - ground_pts[:,2])

pcd_road = pcd_ground.select_by_index(np.where(residuals <= threshold)[0])
pcd_non_road = pcd_ground.select_by_index(np.where(residuals <= threshold)[0], invert=True)

pcd_road.paint_uniform_color([0.0, 1.0, 0.0])
pcd_non_road.paint_uniform_color([0.0, 0, 1.0])

o3d.visualization.draw_geometries(
    [pcd_road, pcd_non_road, pcd_non_ground],
    zoom=0.05,
    front=[-0.1, -0.0, 0.1],
    lookat=[2.1813, 2.0619, 2.0999],
    up=[0.0, -0.0, 1.0],
)