import numpy as np
import os


def fnv_hash_vec(arr):
    """
    FNV64-1A hash for 2D array.
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(
        arr.shape[0], dtype=np.uint64
    )
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def merge_voxel_labels(label_sort, count, desired_label):
    """
    For each voxel (given by contiguous chunks in label_sort,
    with lengths provided in count), check if any label equals desired_label.
    If yes, assign desired_label to the whole voxel.
    Otherwise, assign the majority label.
    """
    num_voxels = count.size
    merged_voxel_labels = np.empty(num_voxels, dtype=np.int32)
    start_idx = 0
    for i in range(num_voxels):
        end_idx = start_idx + count[i]
        voxel_labels = label_sort[start_idx:end_idx]
        if np.any(voxel_labels == desired_label):
            merged_voxel_labels[i] = desired_label
        else:
            # Use majority voting if desired label is not present.
            merged_voxel_labels[i] = 0  # np.bincount(voxel_labels).argmax()
        start_idx = end_idx
    return merged_voxel_labels


seq = "01"
start = 0
end = 1300
path = "dataset/sequences"
label_dir = "labels"
lidar_dir = "velodyne"
grid_size = 0.1

all_pts, all_labels = [], []

odometry = np.load(os.path.join(path, seq, "poses.npy"))

for id in range(start, end + 1):
    pts = np.fromfile(
        os.path.join(path, seq, lidar_dir, str(id).zfill(6) + ".bin"),
        dtype=np.float32,
    ).reshape(-1, 4)
    for i in range(len(pts)):
        pts[i, :3] = (
            odometry[id] @ np.linalg.inv(odometry[start]) @ np.array([*pts[i, :3], 1])
        ).reshape(-1)[:3]
    all_pts.append(pts.copy())

    all_labels.append(
        np.fromfile(
            os.path.join(path, seq, label_dir, str(id).zfill(6) + ".label"),
            dtype=np.int32,
        )
    )

all_pts = np.vstack(all_pts)
all_labels = np.concatenate(all_labels)

scaled_coord = all_pts / np.array(grid_size)
grid_coord = np.floor(scaled_coord).astype(int)
min_coord = grid_coord.min(0)
grid_coord -= min_coord
scaled_coord -= min_coord
min_coord = min_coord * np.array(grid_size)
key = fnv_hash_vec(grid_coord)
idx_sort = np.argsort(key)
key_sort = key[idx_sort]
label_sort = all_labels[idx_sort]
_, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)


desired_label = 40

merged_voxel_labels = merge_voxel_labels(label_sort, count, desired_label)

merged_labels_sorted = merged_voxel_labels[inverse]

merged_labels_original = np.empty_like(merged_labels_sorted)
merged_labels_original[idx_sort] = merged_labels_sorted

for id in range(start, end + 1):
    merged_labels_original[(id - start) * 46080 : (id - start + 1) * 46080].tofile(
        os.path.join(path, seq, "predictions", str(id).zfill(6) + ".label")
    )

# all_pts.reshape(-1).tofile(
#     os.path.join(path, "02", "velodyne", str(0).zfill(6) + ".bin")
# )
# merged_labels_original.tofile(
#     os.path.join(path, "02", "labels", str(0).zfill(6) + ".label")
# )
