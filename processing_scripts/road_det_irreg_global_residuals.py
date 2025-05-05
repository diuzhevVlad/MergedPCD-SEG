from road_det_filt_adv import initialize_model, convert_pc, grid_sample, fit_surface_ransac, draw_points
import open3d as o3d
import cv2
import CSF
import numpy as np
from scipy.spatial import KDTree
from sklearn.preprocessing import PolynomialFeatures
import torch

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.pyplot as plt
from io import BytesIO


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
        
        img_proj = cv2.cvtColor(draw_points(img, road_points[:, :3], np.array(list(map(value_to_colour, residuals[residuals <= threshold])))), cv2.COLOR_RGB2BGR)

        clrbr = colourbar_png(200)
        h, w = clrbr.shape[:2]
        y1, y2 = img_proj.shape[0]-h-10, img_proj.shape[0]-10
        x1, x2 = img_proj.shape[1]-w-10, img_proj.shape[1]-10
        img_proj[y1:y2, x1:x2][(clrbr[:,:,:3]!=255).all(axis=2)] = clrbr[:,:,:3][(clrbr[:,:,:3]!=255).all(axis=2)]

        cv2.imshow("exp", img_proj)
        cv2.waitKey(0)
        return



pcd_files = ["../Huawei/special_cases/flat/000000.bin", "../Huawei/special_cases/pothole/000000.bin", "../Huawei/special_cases/unpaved/000000.bin", "../Huawei/special_cases/speedbump/000000.bin"]
img_files = ["../Huawei/special_cases/flat/000000.png", "../Huawei/special_cases/pothole/000000.png", "../Huawei/special_cases/unpaved/000000.png", "../Huawei/special_cases/speedbump/000000.png"]
cnt = 0
for pcd_file, img_file in zip(pcd_files, img_files):
    procceed(pcd_file, img_file)
