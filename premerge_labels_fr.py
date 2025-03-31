# Install KISS-ICP and Plotting tools
import os

import numpy as np
import tqdm
import torch
from mmdet3d.apis import init_model, inference_segmentor


config_path = "configs/frnet/frnet-semantickitti_seg.py"
checkpoint_path = "frnet-semantickitti_seg.pth"
model = init_model(config_path, checkpoint_path)


data_root = ".."
data_name = "SemanticKITTI"
sequence = "08"
heap_size = 1
sequence_root = os.path.join(data_root, f"{data_name}/dataset/sequences/{sequence}")
save_root = os.path.join(data_root, f"{data_name}/{f'predictions_premerged_fr_{heap_size}' if heap_size > 1 else 'predictions_fr'}/sequences/{sequence}")
merged_root = os.path.join(data_root, f"{data_name}/dataset_premerged_{heap_size}/sequences/{sequence}") if heap_size > 1 else None
merged_root = None
lidar_root = os.path.join(sequence_root, "velodyne")
poses_path = os.path.join(sequence_root, "poses.npy")
pred_root = os.path.join(save_root, "predictions")

poses = np.load(poses_path)
pcd_files = sorted(os.listdir(lidar_root))

learning_map_inv = {
            "ignore_index": -1,  # "unlabeled"
            0: 10,  # "car"
            1: 11,  # "bicycle"
            2: 15,  # "motorcycle"
            3: 18,  # "truck"
            4: 20,  # "other-vehicle"
            5: 30,  # "person"
            6: 31,  # "bicyclist"
            7: 32,  # "motorcyclist"
            8: 40,  # "road"
            9: 44,  # "parking"
            10: 48,  # "sidewalk"
            11: 49,  # "other-ground"
            12: 50,  # "building"
            13: 51,  # "fence"
            14: 70,  # "vegetation"
            15: 71,  # "trunk"
            16: 72,  # "terrain"
            17: 80,  # "pole"
            18: 81,  # "traffic-sign"
        }

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

pcd_heap = []
pcd_lens = []
pcd_ind = []
procceed_full = True
with torch.no_grad():
    for pcd_file_id in tqdm.tqdm(range(len(pcd_files))):
        pts = np.fromfile(
            os.path.join(lidar_root, pcd_files[pcd_file_id]), dtype=np.float32
        ).reshape((-1, 4))

        pts_hom = np.hstack([pts[:, :3], np.ones((pts.shape[0], 1), dtype=pts.dtype)])
        transformed = (
            poses[pcd_file_id]
            @ pts_hom.T
        ).T
        pts[:, :3] = transformed[:, :3]
        pcd_heap.append(pts.copy())
        pcd_lens.append(pts.shape[0])
        pcd_ind.append(pcd_file_id)
        if len(pcd_heap) > heap_size:
            pcd_heap.pop(0)
            pcd_lens.pop(0)
            pcd_ind.pop(0)
        heaped_pts = np.vstack(pcd_heap)
        

        pts_hom = np.hstack([heaped_pts[:, :3], np.ones((heaped_pts.shape[0], 1), dtype=heaped_pts.dtype)])
        transformed = (
            np.linalg.inv(poses[pcd_ind[0]])
            @ pts_hom.T
        ).T
        heaped_pts[:, :3] = transformed[:, :3]

        pts = np.fromfile(
            os.path.join(lidar_root, pcd_files[pcd_file_id]), dtype=np.float32
        ).reshape((-1, 4))

        pts_hom = np.hstack([pts[:, :3], np.ones((pts.shape[0], 1), dtype=pts.dtype)])
        transformed = (
            poses[pcd_file_id]
            @ pts_hom.T
        ).T
        pts[:, :3] = transformed[:, :3]
        pcd_heap.append(pts.copy())
        pcd_lens.append(pts.shape[0])
        pcd_ind.append(pcd_file_id)
        if len(pcd_heap) > heap_size:
            pcd_heap.pop(0)
            pcd_lens.pop(0)
            pcd_ind.pop(0)
        heaped_pts = np.vstack(pcd_heap)
        

        pts_hom = np.hstack([heaped_pts[:, :3], np.ones((heaped_pts.shape[0], 1), dtype=heaped_pts.dtype)])
        transformed = (
            np.linalg.inv(poses[pcd_ind[0]])
            @ pts_hom.T
        ).T
        heaped_pts[:, :3] = transformed[:, :3]

        if procceed_full:
            heaped_pts.reshape(-1).tofile("/tmp/temp_pcd.bin")
            seg_res = inference_segmentor(model, "/tmp/temp_pcd.bin")
            labels = seg_res[0].pred_pts_seg.pts_semantic_mask.cpu().numpy()
            labels = np.vectorize(learning_map_inv.__getitem__)(
                labels & 0xFFFF
            ).astype(np.int32)
            full_labels = labels

        else: 
            feat, grid_coord, min_coord, idx_sort, count, inverse, idx_select = grid_sample(heaped_pts, 0.05)
            feat.reshape(-1).tofile("/tmp/temp_pcd.bin")
            seg_res = inference_segmentor(model, "/tmp/temp_pcd.bin")
            labels = seg_res[0].pred_pts_seg.pts_semantic_mask.cpu().numpy()
            labels = np.vectorize(learning_map_inv.__getitem__)(
                labels & 0xFFFF
            ).astype(np.int32)
            unsorted_inverse = np.empty_like(inverse)
            unsorted_inverse[idx_sort] = inverse
            full_labels = labels[unsorted_inverse]

        if merged_root is not None:
            os.makedirs(os.path.join(merged_root, "velodyne"), exist_ok=True)
            os.makedirs(os.path.join(merged_root, "labels"), exist_ok=True)
            heaped_pts.tofile(os.path.join(merged_root, "velodyne", str(pcd_file_id).zfill(6)+".bin"))
            full_labels.tofile(os.path.join(merged_root, "labels", str(pcd_file_id).zfill(6)+".label"))


        os.makedirs(pred_root, exist_ok=True)
        cs = np.cumsum([0] + pcd_lens)
        full_labels[cs[len(pcd_heap)//2]:cs[len(pcd_heap)//2+1]].tofile(os.path.join(pred_root, str(pcd_ind[len(pcd_heap)//2]).zfill(6)+".label"))
        if pcd_file_id == len(pcd_files) - 1:
            for i in range(len(pcd_heap)//2, heap_size):
                full_labels[cs[i]:cs[1+i]].tofile(os.path.join(pred_root, str(pcd_ind[i]).zfill(6)+".label"))
        
        del heaped_pts, feat
        torch.cuda.empty_cache()
os.remove("/tmp/temp_pcd.bin")
