import os
import numpy as np
import tqdm
import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from torchpack.utils.config import configs
from core import builder
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate


configs.load("configs/semantic_kitti/spvcnn/cr0p5.yaml", recursive=True)

model = builder.make_model().cuda()
checkpoint = torch.load(
    "runs/run-993f3583/checkpoints/max-iou-test.pt", map_location=lambda storage, loc: storage.cuda()
)
load_state_info = model.load_state_dict(checkpoint["model"], strict=True)
assert load_state_info

data_root = ".."
data_name = "SemanticKITTI"
sequence = "08"
heap_size = 3
sequence_root = os.path.join(data_root, f"{data_name}/dataset/sequences/{sequence}")
save_root = os.path.join(data_root, f"{data_name}/{f'predictions_premerged_pt_{heap_size}' if heap_size > 1 else 'dataset'}/sequences/{sequence}")
merged_root = os.path.join(data_root, f"{data_name}/dataset_premerged_{heap_size}/sequences/{sequence}") if heap_size > 1 else None
lidar_root = os.path.join(sequence_root, "velodyne")
poses_path = os.path.join(sequence_root, "poses.npy")
pred_root = os.path.join(save_root, "predictions")

pcd_files = sorted(os.listdir(lidar_root))
poses = np.load(poses_path)

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

pcd_heap = []
pcd_lens = []
pcd_ind = []
procceed_full = False
model.eval()
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

        if procceed_full:
            coords = np.round(heaped_pts[:, :3] / 0.05)
            coords -= coords.min(0, keepdims=1)
            feats = heaped_pts
            # coords, indices, inverse = sparse_quantize(coords, return_index=True, return_inverse=True)
            coords = torch.tensor(coords, dtype=torch.int)
            feats = torch.tensor(feats, dtype=torch.float)
            inputs = SparseTensor(coords=coords, feats=feats)
            inputs = sparse_collate([inputs]).cuda()
            outputs = model(inputs)
            outputs = outputs.argmax(1).cpu().numpy()
            outputs = np.vectorize(learning_map_inv.__getitem__)(
                                outputs & 0xFFFF
                            ).astype(np.int32)
            full_labels = outputs

        else: 
            coords = np.round(heaped_pts[:, :3] / 0.05)
            coords -= coords.min(0, keepdims=1)
            feats = heaped_pts
            coords, indices, inverse = sparse_quantize(coords, return_index=True, return_inverse=True)
            coords = torch.tensor(coords, dtype=torch.int)
            feats = torch.tensor(feats[indices], dtype=torch.float)
            inputs = SparseTensor(coords=coords, feats=feats)
            inputs = sparse_collate([inputs]).cuda()
            outputs = model(inputs)
            outputs = outputs.argmax(1).cpu().numpy()
            outputs = outputs[inverse]
            outputs = np.vectorize(learning_map_inv.__getitem__)(
                                outputs & 0xFFFF
                            ).astype(np.int32)
            full_labels = outputs

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
        
        torch.cuda.empty_cache()
