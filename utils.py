import os
from collections import OrderedDict
import CSF
import cv2
import numpy as np
import torch
import tqdm
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from typing import Tuple
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.pyplot as plt
from io import BytesIO

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

def draw_points(img, pts, clr, extr, proj_mat):
    points_hom = np.hstack([pts, np.ones((pts.shape[0],1))])
    points_cam = extr @ points_hom.T

    # Project points to image plane
    uv, _, valid_indices = project_points_to_camera(points_cam, proj_mat, (1280, 720))
    im_cp = img.copy()
    overlay = img.copy()

    for point, d in zip(uv.T, clr[valid_indices]):
        c = d * 255
        c = (int(c[0]), int(c[1]), int(c[2]))
        cv2.circle(overlay, point, radius=2, color=c, thickness=cv2.FILLED)
    
    return cv2.addWeighted(overlay, 0.4, im_cp, 1 - 0.4, 0)

def value_to_colour(value, vmin=0, vmax=0.06, cmap_name="plasma"):
    # Normalise value → [0, 1] and clip
    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    # Grab the chosen colormap
    cmap   = cm.get_cmap(cmap_name)
    # Convert to RGBA
    rgba   = cmap(norm(value))
    return rgba[:3]

def colourbar_png(height, vmin=0, vmax=0.06, cmap="plasma", label=None):
    fig, ax = plt.subplots(figsize=(1.4, height / 100), dpi=100)
    fig.subplots_adjust(left=0.4, right=0.7, top=0.98, bottom=0.02)
    ax.tick_params(axis='y', colors='#C0C0C0')

    norm = mcolors.Normalize(vmin, vmax)
    cb   = mcolorbar.ColorbarBase(ax, cmap=cmap, norm=norm,
                                     orientation="vertical")
    if label:
        cb.set_label(label, rotation=90, labelpad=12)

    buf = BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)
    bar = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    return cv2.imdecode(bar, cv2.IMREAD_UNCHANGED)