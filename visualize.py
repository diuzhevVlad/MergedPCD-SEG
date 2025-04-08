import rerun as rr
import numpy as np
import matplotlib.pyplot as plt

rr.init("rerun_example_my_data", spawn=True)

base = plt.get_cmap("hsv")
colors = []
for i in range(256):
    x = (i / 255) * 0.8
    colors.append(base(x))
colors = np.array(colors)

lidar_dir = "../Huawei/dataset/sequences/03/velodyne"
labels_dir = "../Huawei/dataset/sequences/03/predictions"

pts = np.fromfile(f"{lidar_dir}/000000.bin", dtype=np.float32).reshape(-1, 4)

print(pts[:,3].max())
# labels = np.fromfile(f"{labels_dir}/000000.label", dtype=np.int32)
# pts_cl = colors[pts[:,3].astype(int)]

# rr.log(
#     "my_points",
#     rr.Points3D(pts[:,:3], colors=pts_cl, radii=0.05, labels=labels.astype(np.bytes0))
# )