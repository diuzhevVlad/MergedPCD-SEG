import os
from collections import OrderedDict
import CSF
import cv2
import numpy as np
import rerun as rr
import torch
import tqdm
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from typing import Tuple
from model import PointTransformerV3
import open3d as o3d


# Constants
LINES = 96 # 96 / 192 
# MAIN_DIR = "/home/vladislav/Documents/Huawei/rosbag2_2025_04_08-15_42_42__unparsed"
# MAIN_DIR = "/home/vladislav/Documents/Huawei/rosbag2_2025_03_04-20_38_39__unparsed"
# MAIN_DIR = "/home/vladislav/Documents/Huawei/rosbag2_2025_03_04-20_23_54__unparsed"
# MAIN_DIR = "/home/vladislav/Documents/Huawei/rosbag2_2025_02_28-16_28_54__unparsed"
# MAIN_DIR = "/home/vladislav/Documents/Huawei/rosbag2_2025_02_21-16_27_56__unparsed"
# MAIN_DIR = "/home/vladislav/Documents/Huawei/rosbag2_2025_02_28-16_45_00__unparsed"
MAIN_DIR = "/home/vladislav/Documents/Huawei/rosbag2_2025_02_28-16_36_19__unparsed"
# MAIN_DIR = "/home/vladislav/Documents/Huawei/rosbag2_2025_02_21-16_54_13__unparsed"
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


def initialize_model():
    """Initialize and load the segmentation model."""
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
    checkpoint = torch.load("best_model.pth", map_location=lambda storage, loc: storage.cuda())

    # Load model weights
    weight_backbone = OrderedDict()
    weight_seg_head = OrderedDict()

    for key, value in checkpoint.items():
        if "backbone" in key:
            weight_backbone[key.replace("module.backbone.", "")] = value
        elif "seg_head" in key:
            weight_seg_head[key.replace("module.seg_head.", "")] = value

    segmodel.load_state_dict(weight_backbone, strict=True)
    seg_head.load_state_dict(weight_seg_head, strict=True)

    return segmodel, seg_head

def project_points_to_camera(
    points: np.ndarray, proj_matrix: np.ndarray, cam_res: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # Return original valid indices
    if points.shape[0] == 3:
        points = np.vstack((points, np.ones((1, points.shape[1]))))
    in_image = points[2, :] > 0  # Initial filter for points in front of the camera
    depths = points[2, in_image]
    uvw = np.dot(proj_matrix, points[:, in_image])
    uv = uvw[:2, :]
    w = uvw[2, :]
    uv[0, :] /= w
    uv[1, :] /= w
    valid = (uv[0, :] >= 0) & (uv[0, :] < cam_res[0]) & (uv[1, :] >= 0) & (uv[1, :] < cam_res[1])
    uv_valid = uv[:, valid].astype(int)
    depths_valid = depths[valid]
    # Get indices of original points that are valid
    original_indices = np.where(in_image)[0][valid]
    return uv_valid, depths_valid, original_indices


def fit_surface_ransac(points, order, distance_threshold=0.05):
    """Fit a polynomial surface using RANSAC and return residuals."""
    X = points[:, :2]  # XY coordinates
    y = points[:, 2]   # Z (height)

    poly = PolynomialFeatures(degree=order, include_bias=True)
    X_poly = poly.fit_transform(X)

    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        residual_threshold=distance_threshold,
        max_trials=100
    )
    ransac.fit(X_poly, y)

    y_pred = ransac.predict(X_poly)
    residuals = np.abs(y - y_pred)

    return residuals, ransac


def fnv_hash_vec(arr):
    """FNV64-1A hash for a 2D array."""
    assert arr.ndim == 2
    arr = arr.copy().astype(np.uint64)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    
    return hashed_arr


def grid_sample(data, grid_size):
    """Sample points using grid-based approach."""
    scaled_coord = data[:, :3] / np.array(grid_size)
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
        np.cumsum(np.insert(count, 0, 0)[0:-1]) + 
        np.random.randint(0, count.max(), count.size) % count
    )
    idx_unique = idx_sort[idx_select]
    
    return (
        data[idx_unique], grid_coord[idx_unique], 
        min_coord.reshape([1, 3]), idx_sort, 
        count, inverse, idx_select
    )

def convert_pc(point_cloud: np.ndarray, 
               elev_range, 
               azim_range,
               num_elev: int, 
               num_azim: int = 480) -> np.ndarray:
    assert point_cloud.shape[1] == 3, f'Expected pc shape to be (n, 3), got {point_cloud.shape}'    
    point_cloud = point_cloud.reshape(num_azim, num_elev, 3)
    neg_mask = np.where(point_cloud[:, :, 0] >= 0, 1, -1)
    distance_matrix = (
        np.sqrt(point_cloud[:, :, 0] ** 2 + point_cloud[:, :, 1] ** 2 + point_cloud[:, :, 2] ** 2)
        * neg_mask  # keep negative values
    )
    elev_step = (elev_range[1] - elev_range[0]) / num_elev
    azim_step = (azim_range[1] - azim_range[0]) / num_azim
    elev = np.arange(elev_range[0], elev_range[1], elev_step)[::-1]
    azim = np.arange(azim_range[0], azim_range[1], azim_step)[::-1]
    elev_mat, azim_mat = np.meshgrid(np.deg2rad(elev), np.deg2rad(azim))
    x = (distance_matrix * np.cos(elev_mat) * np.cos(azim_mat)).flatten()
    y = (distance_matrix * np.cos(elev_mat) * np.sin(azim_mat)).flatten()
    z = (distance_matrix * np.sin(elev_mat)).flatten()
    
    return np.stack((x, y, z), axis=-1).reshape(-1, 3)



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

def draw_points(img, pts, clr):
    points_hom = np.hstack([pts, np.ones((pts.shape[0],1))])
    points_cam = EXTRINSIC @ points_hom.T

    # Project points to image plane
    uv, _, valid_indices = project_points_to_camera(points_cam, PROJ_MAT, (1280, 720))
    im_cp = img.copy()
    overlay = img.copy()

    for point, d in zip(uv.T, clr[valid_indices]):
        c = d * 255
        c = (int(c[0]), int(c[1]), int(c[2]))
        cv2.circle(overlay, point, radius=2, color=c, thickness=cv2.FILLED)
    
    return cv2.addWeighted(overlay, 0.4, im_cp, 1 - 0.4, 0)


def main():
    """Main processing loop."""
    rr.init("rerun_road_segmentation", spawn=True)

    segmodel, seg_head = initialize_model()

    pcd_files = sorted(os.listdir(LIDAR_DIR))
    img_files = sorted(os.listdir(IMG_DIR))
    t = 0

    start = 0 if START is None else START
    end = len(pcd_files) if STOP is None else STOP

    for pcd_file, img_file in tqdm.tqdm(zip(pcd_files[start:end], img_files[start:end])):
        img, all_pts, all_clr, _, road_pts_num = process_frame(pcd_file, img_file, segmodel, seg_head)
        img_proj = draw_points(img, all_pts[:road_pts_num, :3], all_clr[:road_pts_num])
        
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