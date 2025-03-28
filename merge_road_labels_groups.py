import numpy as np
import os
import tqdm

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

def merge_voxel_labels(label_sort, count, desired_label):
    """
    For each voxel (given by contiguous chunks in label_sort with counts),
    perform majority voting.
    """
    num_voxels = count.size
    merged_voxel_labels = np.empty(num_voxels, dtype=np.int32)
    start_idx = 0
    for i in range(num_voxels):
        end_idx = start_idx + count[i]
        voxel_labels = label_sort[start_idx:end_idx]
        merged_voxel_labels[i] = np.bincount(voxel_labels).argmax()
        start_idx = end_idx
    return merged_voxel_labels

def clamp(val, low, high):
    return max(low, min(val, high))

# ------------------------
# Parameters and paths
seq = "02"
path = "../Huawei/dataset/sequences"
save_path = "../Huawei/predictions_postmerged/sequences"
label_dir = "predictions_norm"
lidar_dir = "velodyne"
frame_files = os.listdir(os.path.join(path, seq, lidar_dir))
frame_num = len(frame_files)
start_frame = 0
end_frame = frame_num - 1  # inclusive

grid_size = 0.1
group_size = 5  # sliding window size
desired_label = 40  # overriding label

# Load odometry and select the first frame as a fixed base.
odometry = np.load(os.path.join(path, seq, "poses.npy"))
base_pose = odometry[start_frame]
num_frames = end_frame - start_frame + 1

# Preload and transform each frame’s point cloud.
# Transformation is vectorized by working in homogeneous coordinates.
all_frames_pts = []
all_frames_labels = []
frame_point_counts = []

print("Loading and transforming frames...")
for frame_id in tqdm.tqdm(range(start_frame, end_frame + 1), desc="Frames"):
    pts_file = os.path.join(path, seq, lidar_dir, str(frame_id).zfill(6) + ".bin")
    pts = np.fromfile(pts_file, dtype=np.float32).reshape(-1, 4)
    # Create homogeneous coordinates and transform all points at once.
    pts_hom = np.hstack([pts[:, :3], np.ones((pts.shape[0], 1), dtype=pts.dtype)])
    transformed = (odometry[frame_id] @ np.linalg.inv(base_pose) @ pts_hom.T).T
    pts[:, :3] = transformed[:, :3]
    all_frames_pts.append(pts.copy())

    label_file = os.path.join(path, seq, label_dir, str(frame_id).zfill(6) + ".label")
    labels = np.fromfile(label_file, dtype=np.int32)
    all_frames_labels.append(labels)

    frame_point_counts.append(pts.shape[0])

# Prepare container for per-frame window predictions.
# predictions_per_frame[i] will be a list of tuples: (window_start, prediction)
predictions_per_frame = [[] for _ in range(num_frames)]

# Process overlapping sliding windows sequentially.
print("Processing sliding windows sequentially...")
for window_start in tqdm.tqdm(range(0, num_frames - group_size + 1), desc="Windows"):
    window_end = window_start + group_size  # exclusive window end
    window_pts_list = all_frames_pts[window_start:window_end]
    window_labels_list = all_frames_labels[window_start:window_end]
    window_point_counts = [pts.shape[0] for pts in window_pts_list]

    # Concatenate points and labels from the window.
    window_pts = np.vstack(window_pts_list)
    window_labels = np.concatenate(window_labels_list)

    # Voxelization: compute voxel grid indices.
    scaled_coord = window_pts / grid_size
    grid_coord = np.floor(scaled_coord).astype(np.int32)
    min_coord = grid_coord.min(axis=0)
    grid_coord -= min_coord  # shift so indices are non-negative

    # Compute hash keys for each voxel.
    key = fnv_hash_vec(grid_coord)
    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    label_sort = window_labels[idx_sort]

    # Group voxels using unique keys.
    _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

    # Merge labels per voxel.
    merged_voxel_labels = merge_voxel_labels(label_sort, count, desired_label)

    # Map merged voxel labels back to the original order.
    merged_labels_sorted = merged_voxel_labels[inverse]
    merged_labels_window = np.empty_like(merged_labels_sorted)
    merged_labels_window[idx_sort] = merged_labels_sorted

    # Split the merged labels to obtain per-frame predictions.
    start_idx = 0
    for i in range(group_size):
        num_points = window_point_counts[i]
        frame_prediction = merged_labels_window[start_idx:start_idx + num_points]
        predictions_per_frame[window_start + i].append((window_start, frame_prediction))
        start_idx += num_points

# Unmerge the results: for each frame, select one prediction (from the window where the frame is most centered)
half_window = group_size // 2
final_predictions = []

for i in range(num_frames):
    desired_ws = clamp(i - half_window, 0, num_frames - group_size)
    best_pred = None
    best_diff = None
    for ws, pred in predictions_per_frame[i]:
        diff = abs(ws - desired_ws)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_pred = pred
    final_predictions.append(best_pred)

# ------------------------
# Save final predictions to disk.
predictions_dir = os.path.join(save_path, seq, "predictions_norm")
os.makedirs(predictions_dir, exist_ok=True)
print("Saving predictions...")
for i in range(num_frames):
    output_file = os.path.join(predictions_dir, str(start_frame + i).zfill(6) + ".label")
    final_predictions[i].tofile(output_file)
