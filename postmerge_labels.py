import numpy as np
import os
import tqdm

def fnv_hash_vec(arr):
    assert arr.ndim == 2
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr

def merge_voxel_labels(label_sort, count, desired_label):
    num_voxels = count.size
    merged_voxel_labels = np.empty(num_voxels, dtype=np.int32)
    start_idx = 0
    for i in range(num_voxels):
        end_idx = start_idx + count[i]
        voxel_labels = label_sort[start_idx:end_idx]
        if desired_label is not None and np.any(voxel_labels == desired_label):
            merged_voxel_labels[i] = desired_label
        else:
            merged_voxel_labels[i] = np.bincount(voxel_labels).argmax()
        start_idx = end_idx
    return merged_voxel_labels


data_root = ".."
data_name = "SemanticKITTI"
labels_name = "predictions_sp" # dataset
sequence = "08"
heap_size = 3
sequence_root = os.path.join(data_root, f"{data_name}/dataset/sequences/{sequence}")
labels_root = os.path.join(data_root, f"{data_name}/{labels_name}/sequences/{sequence}/predictions")
save_root = os.path.join(data_root, f"{data_name}/{f'predictions_postmerged_sp_{heap_size}'}/sequences/{sequence}")
merged_root = os.path.join(data_root, f"{data_name}/dataset_postmerged_{heap_size}/sequences/{sequence}")
merged_root = None
lidar_root = os.path.join(sequence_root, "velodyne")
poses_path = os.path.join(sequence_root, "poses.npy")
pred_root = os.path.join(save_root, "predictions")

grid_size = 0.1
desired_label = None  # overriding label

odometry = np.load(poses_path)
base_pose = odometry[0]

pcd_heap = []
label_heap = []
pcd_lens = []
pcd_ind = []
heap_size = 5
print("Loading and transforming frames...")
for frame_id in tqdm.tqdm(range(odometry.shape[0]), desc="Frames"):
    pts_file = os.path.join(lidar_root, str(frame_id).zfill(6) + ".bin")
    pts = np.fromfile(pts_file, dtype=np.float32).reshape(-1, 4)
    pts_hom = np.hstack([pts[:, :3], np.ones((pts.shape[0], 1), dtype=pts.dtype)])
    transformed = (odometry[frame_id] @ pts_hom.T).T
    pts[:, :3] = transformed[:, :3]

    label_file = os.path.join(labels_root, str(frame_id).zfill(6) + ".label")
    labels = np.fromfile(label_file, dtype=np.int32)

    pcd_heap.append(pts.copy())
    label_heap.append(labels.copy())
    pcd_lens.append(pts.shape[0])
    pcd_ind.append(frame_id)
    if len(pcd_heap) > heap_size:
        pcd_heap.pop(0)
        label_heap.pop(0)
        pcd_lens.pop(0)
        pcd_ind.pop(0)

    heaped_pts = np.vstack(pcd_heap)
    heaped_labels = np.concatenate(label_heap)

    scaled_coord = heaped_pts / grid_size
    grid_coord = np.floor(scaled_coord).astype(np.int32)
    min_coord = grid_coord.min(axis=0)
    grid_coord -= min_coord  # shift so indices are non-negative

    # Compute hash keys for each voxel.
    key = fnv_hash_vec(grid_coord)
    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    label_sort = heaped_labels[idx_sort]

    # Group voxels using unique keys.
    _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

    # Merge labels per voxel.
    merged_voxel_labels = merge_voxel_labels(label_sort, count, desired_label)

    # Map merged voxel labels back to the original order.
    merged_labels_sorted = merged_voxel_labels[inverse]
    merged_labels_heap = np.empty_like(merged_labels_sorted)
    merged_labels_heap[idx_sort] = merged_labels_sorted

    # print(np.count_nonzero(heaped_labels == merged_labels_heap) / heaped_labels.shape[0])

    if merged_root is not None:
        os.makedirs(os.path.join(merged_root, "velodyne"), exist_ok=True)
        os.makedirs(os.path.join(merged_root, "labels"), exist_ok=True)
        heaped_pts.tofile(os.path.join(merged_root, "velodyne", str(frame_id).zfill(6)+".bin"))
        merged_labels_heap.tofile(os.path.join(merged_root, "labels", str(frame_id).zfill(6)+".label"))

    os.makedirs(pred_root, exist_ok=True)
    cs = np.cumsum([0] + pcd_lens)
    merged_labels_heap[cs[len(pcd_heap)//2]:cs[len(pcd_heap)//2+1]].tofile(os.path.join(pred_root, str(pcd_ind[len(pcd_heap)//2]).zfill(6)+".label"))
    if frame_id == odometry.shape[0] - 1:
        for i in range(len(pcd_heap)//2, heap_size):
            merged_labels_heap[cs[i]:cs[1+i]].tofile(os.path.join(pred_root, str(pcd_ind[i]).zfill(6)+".label"))


