from road_det_filt_adv import initialize_model, convert_pc, grid_sample, fit_surface_ransac, draw_points
import cv2
import CSF
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import torch
import rerun as rr


import matplotlib as mpl
mpl.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.pyplot as plt
from io import BytesIO

import numpy as np
from scipy.interpolate import griddata
from sklearn.linear_model import LinearRegression, RANSACRegressor

# ────────────────────────────────────────────────────────────────────────
def residuals_by_ransac_region(pts, grid_size=0.5,
                               min_points=20,
                               ransac_thresh=0.03,
                               max_trials=1000,
                               return_planes=False):
    """
    Fit a plane with RANSAC in every grid cell and give per-point residuals.

    Parameters
    ----------
    pts : (N, 3) float array   -- scattered [x, y, z] points
    grid_size : float          -- square cell size in metres
    min_points : int           -- skip cells with fewer points than this
    ransac_thresh : float      -- residual threshold for RANSAC (metres)
    max_trials : int           -- max RANSAC iterations per cell
    return_planes : bool       -- also return (ny-1, nx-1, 3) plane tensor

    Returns
    -------
    residuals : (N,) float     -- z_meas − z_plane  (NaN if no plane)
    planes    : (ny-1, nx-1, 3) float (optional)   -- (a, b, c) per cell
    """
    # 1. build regular XY grid -------------------------------------------
    xmin, ymin = pts[:, :2].min(axis=0)
    xmax, ymax = pts[:, :2].max(axis=0)
    xi = np.arange(xmin, xmax + grid_size, grid_size)
    yi = np.arange(ymin, ymax + grid_size, grid_size)
    nx, ny = len(xi) - 1, len(yi) - 1

    planes = np.full((ny, nx, 3), np.nan, dtype=np.float32)
    residuals = np.full(len(pts), np.nan, dtype=np.float32)

    # 2. map every point to its cell -------------------------------------
    idx_x = np.floor((pts[:, 0] - xmin) / grid_size).astype(int)
    idx_y = np.floor((pts[:, 1] - ymin) / grid_size).astype(int)

    # clip points on the max edge into the last valid cell
    idx_x[idx_x == nx] = nx - 1
    idx_y[idx_y == ny] = ny - 1

    # group points by (idx_y, idx_x) pair using a dict
    from collections import defaultdict
    bins = defaultdict(list)
    for i, (ix, iy) in enumerate(zip(idx_x, idx_y)):
        bins[(iy, ix)].append(i)

    # 3. plane-fit each populated cell -----------------------------------
    lr = LinearRegression()

    for (iy, ix), pt_idx in bins.items():
        if len(pt_idx) < min_points:
            continue

        P = pts[pt_idx]
        X, y = P[:, :2], P[:, 2]

        ransac = RANSACRegressor(
            lr,
            residual_threshold=ransac_thresh,
            min_samples=max(min_points, int(0.5 * len(pt_idx))),
            max_trials=max_trials,
            random_state=0,
        )
        try:
            ransac.fit(X, y)
        except ValueError:  # degenerate cell (all points collinear, etc.)
            continue

        a, b = ransac.estimator_.coef_
        c     = ransac.estimator_.intercept_
        planes[iy, ix] = (a, b, c)

        z_pred = a * X[:, 0] + b * X[:, 1] + c
        residuals[pt_idx] = y - z_pred  # signed residual

    return (residuals, planes) if return_planes else residuals


def _fit_plane(pts3):
    """Return (a, b, c) for z = a·x + b·y + c through ≥3 points."""
    A = np.c_[pts3[:, 0], pts3[:, 1], np.ones(len(pts3))]
    coef, *_ = np.linalg.lstsq(A, pts3[:, 2], rcond=None)
    return coef          # (a, b, c)


def compute_residuals_pointwise(pts, grid_size=0.5, fill_value=np.nan):
    # 1. Build XY grid ----------------------------------------------------
    xmin, ymin = pts[:, :2].min(axis=0)
    xmax, ymax = pts[:, :2].max(axis=0)
    xi = np.arange(xmin, xmax + grid_size, grid_size)
    yi = np.arange(ymin, ymax + grid_size, grid_size)
    grid_x, grid_y = np.meshgrid(xi, yi)             # ny × nx

    # 2. Interpolate node heights (linear) -------------------------------
    grid_z = griddata(pts[:, :2], pts[:, 2],
                      xi=(grid_x, grid_y),
                      method="linear",
                      fill_value=fill_value)

    ny, nx  = grid_x.shape
    planes  = np.full((ny - 1, nx - 1, 3), np.nan, dtype=np.float32)

    # 3. Fit a tiny plane per quad (skip if any corner is NaN) -----------
    for j in range(ny - 1):
        for i in range(nx - 1):
            quad_z = grid_z[j:j+2, i:i+2].ravel()
            if np.isnan(quad_z).any():
                continue
            pts3 = np.array([[grid_x[j,   i],   grid_y[j,   i],   quad_z[0]],
                             [grid_x[j,   i+1], grid_y[j,   i+1], quad_z[1]],
                             [grid_x[j+1, i],   grid_y[j+1, i],   quad_z[2]]])
            planes[j, i] = _fit_plane(pts3)

    # 4. Compute residual for every point --------------------------------
    residuals = np.full(len(pts), np.nan, dtype=np.float32)

    ix = np.floor((pts[:, 0] - grid_x[0, 0]) / grid_size).astype(int)
    iy = np.floor((pts[:, 1] - grid_y[0, 0]) / grid_size).astype(int)

    good = (ix >= 0) & (ix < nx-1) & (iy >= 0) & (iy < ny-1) & \
           ~np.isnan(planes[iy, ix, 0])

    a, b, c = planes[iy[good], ix[good]].T
    z_pred  = a*pts[good, 0] + b*pts[good, 1] + c
    residuals[good] = pts[good, 2] - z_pred

    return residuals


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
        residuals, ransac_model = fit_surface_ransac(road_points[:, :3], order=1)
        threshold = np.percentile(residuals, 95)

        # Recompute residuals for all ground points
        poly = PolynomialFeatures(degree=1, include_bias=True)
        X_poly = poly.fit_transform(ground_pts[:, :2])
        Y_poly = ransac_model.predict(X_poly)
        residuals = np.abs(Y_poly - ground_pts[:, 2])

        # Prepare data for visualization
        road_points = ground_pts[residuals <= threshold]
        non_road_points = ground_pts[residuals > threshold]
        
        # img_proj = draw_points(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), road_points[:, :3], np.array(list(map(value_to_colour, residuals[residuals <= threshold]))))
        # img_proj = draw_points(img, road_points[:, :3], np.array([[0.,0.,255.]]*np.count_nonzero(road_points)))
        # img_proj = draw_points(img_proj, non_road_points[:, :3], np.array([[255.,0.,0.]]*len(non_road_points)))

        # clrbr = colourbar_png(200)
        # h, w = clrbr.shape[:2]

        # # ROI where we’ll paste the bar (bottom-right corner)
        # y1, y2 = img_proj.shape[0]-h-10, img_proj.shape[0]-10
        # x1, x2 = img_proj.shape[1]-w-10, img_proj.shape[1]-10
        # img_proj[y1:y2, x1:x2][(clrbr[:,:,:3]!=255).all(axis=2)] = clrbr[:,:,:3][(clrbr[:,:,:3]!=255).all(axis=2)]

        # cv2.imshow("exp", img_proj)
        # cv2.waitKey(0)
        # return

    resid, cell_planes = residuals_by_ransac_region(
        road_points[:,:3],            # point set
        grid_size=2.,      # 25 cm cells
        ransac_thresh=0.02,  # ≤2 cm counted as inlier
        return_planes=True
    )
    colors = np.array(list(map(value_to_colour, np.abs(resid))))


    img_proj = cv2.cvtColor(draw_points(img, road_points[:, :3], colors), cv2.COLOR_RGB2BGR)
    clrbr = colourbar_png(200)
    h, w = clrbr.shape[:2]
    y1, y2 = img_proj.shape[0]-h-10, img_proj.shape[0]-10
    x1, x2 = img_proj.shape[1]-w-10, img_proj.shape[1]-10
    img_proj[y1:y2, x1:x2][(clrbr[:,:,:3]!=255).all(axis=2)] = clrbr[:,:,:3][(clrbr[:,:,:3]!=255).all(axis=2)]


    cv2.imshow("exp", img_proj)
    cv2.waitKey(0)


pcd_files = ["../Huawei/special_cases/flat/000000.bin", "../Huawei/special_cases/pothole/000000.bin", "../Huawei/special_cases/unpaved/000000.bin", "../Huawei/special_cases/speedbump/000000.bin"]
img_files = ["../Huawei/special_cases/flat/000000.png", "../Huawei/special_cases/pothole/000000.png", "../Huawei/special_cases/unpaved/000000.png", "../Huawei/special_cases/speedbump/000000.png"]
cnt = 0
for pcd_file, img_file in zip(pcd_files, img_files):
    procceed(pcd_file, img_file)
