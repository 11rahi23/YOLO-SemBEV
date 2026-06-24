import os
import sys

# Add conda environment's DLL path
if 'CONDA_PREFIX' in os.environ:
    conda_bin = os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'bin')
    if os.path.exists(conda_bin):
        os.add_dll_directory(conda_bin)
        print(f"Added DLL directory: {conda_bin}")
else:
    # Hardcode path if CONDA_PREFIX not set
    conda_bin = r"C:\Users\arahi\AppData\Local\miniconda3\envs\radar_fusion\Library\bin"
    if os.path.exists(conda_bin):
        os.add_dll_directory(conda_bin)
        print(f"Added DLL directory: {conda_bin}")

import torch
from torch.utils.data import Dataset
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import os
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm.auto import tqdm
import random
from keras.models import load_model
import segmentation_models as sm
import tensorflow as tf

sm.set_framework('tf.keras')

print(tf.config.list_physical_devices('GPU'))

class NuscDataset(Dataset):
    def __init__(self, image_size, path = 'v1.0-mini', max_samples=None, save_raw_img=False):
        super().__init__()

        self.mode = 'train'
        self.x_train_images, self.x_train_radar, self.x_train_camK, self.x_train_seg, self.x_val_images, self.x_val_radar, self.x_val_camK, self.x_val_seg, self.y_train, self.y_val = None, None, None, None, None, None, None, None, None, None
        self.image_paths = []
        self.image_size = image_size
        self.radar_pcd, self.cam_intrinsics, self.labels = [], [], []
        self.untransformed_labels = []
        #self.raw_images = []
        self.time_threshold = 100000
        self.cam_channel = 'CAM_FRONT'
        self.radar_channel = 'RADAR_FRONT'
        self.classes = ['animal', 'human', 'stroller/wheelchair', 'object', 'vehicle' ]

        self.path = path
        self.nusc = NuScenes(version = 'v1.0-trainval', dataroot=path, verbose=True)
        
        self.max_samples = max_samples
        all_samples = self.nusc.sample

        # Segmentation parameters
        self.seg_model = load_model('segmented_cityscape_resnet34_unet.h5')
        self.seg_input_h, self.seg_input_w = 480, 640
        output_shape = self.seg_model.output_shape
        self.num_cityscape_classes = output_shape[3]

        self.class_mapping = self.create_cityscapes_to_simplified_mapping()
        self.num_seg_classes = 12

        print ("Loaded Cityscape segmentation model: ")
        print(f"Original classes: {self.num_cityscape_classes}")
        print(f"Simplified classes: {self.num_seg_classes}")

        #DBSCAN Cluster Parameters
        self.Z = 100
        self.X = 100*np.tan(60*np.pi/180)
        self.Y = self.Z*np.tan(9*np.pi/180)
        self.min_pts = 3
        self.eps = 5

        # Camera visibility settings
        self.camZ = 100
        self.camX = 50
        self.camY = 10
        cam_sd = None
        i=0

        if self.max_samples is not None and self.max_samples <len(all_samples):
            samples_to_process = random.sample(all_samples, self.max_samples)
        else:
            samples_to_process = all_samples
            
        for sample in tqdm(samples_to_process):
            #print("sample: ", sample)
            #print("sample id: ", i)
            cam_token = sample['data'][self.cam_channel]
            radar_token = sample['data'][self.radar_channel]
            current_cam = self.nusc.get('sample_data', cam_token)
            current_radar = self.nusc.get('sample_data', radar_token)
            #print("Current camera: ", current_cam['filename'])
            #print("Current_radar: ", current_radar)

            cam_calib = self.nusc.get('calibrated_sensor', current_cam['calibrated_sensor_token'])
            cam_intrinsic = np.array(cam_calib['camera_intrinsic'])
            cam_ego_pose = self.nusc.get('ego_pose', current_cam['ego_pose_token'])
            self.cam_intrinsics.append(cam_intrinsic)

            # =============================== Store path only =======================================
            cam_path = os.path.join(self.nusc.dataroot, current_cam['filename'])
            self.image_paths.append(cam_path)

             
            cam_timestamp = current_cam['timestamp']
            closest_radar = None
            closest_radars = []
            time_diffs = []
            min_time_diff = float('inf')
            while current_radar is not None:
                time_diff = abs(current_radar['timestamp'] - cam_timestamp)
                #print("Time diff (Unchecked): ", time_diff)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_radar = current_radar
                if time_diff <self.time_threshold:
                    #print("Time diff (after threshold): ", time_diff)
                    closest_radars.append(current_radar)
                    time_diffs.append(time_diff)
                
                if current_radar['next'] != '':
                    current_radar = self.nusc.get('sample_data', current_radar['next'])
                else:
                    break
            
            #print("Closest radars: ", len(closest_radars))

            radar_calib = self.nusc.get('calibrated_sensor', closest_radar['calibrated_sensor_token'])
            radar_ego_pose = self.nusc.get('ego_pose', closest_radar['ego_pose_token'])

            #transform closest radar point to camera frame

            closest_radar_point = self.extract_radar_data(closest_radar)
            #print("point before transform: ", closest_radar_point[0])
            points_img, closest_radar_point_cam = self.transform_radar_to_camera(closest_radar_point, radar_calib, radar_ego_pose, cam_calib, cam_ego_pose, cam_intrinsic)
            #print("point in image plane ", points_img[0])
            #print("transformed point in 3d: ", points_in_cam[0])
            #print("Closest point (key point) shape: ", closest_radar_point.shape)
            #print("closest point (key point): ", closest_radar_point)

            closest_radar_data = []

            #Stacking x,y,z of closest points for clustering
            for point in closest_radars:
                radar_from_point = self.extract_radar_data(point)
                calib = self.nusc.get('calibrated_sensor', point['calibrated_sensor_token'])
                ego_pose = self.nusc.get('ego_pose', point['ego_pose_token'])
                _, converted_radar_from_point = self.transform_radar_to_camera(radar_from_point, calib, ego_pose, cam_calib, cam_ego_pose,cam_intrinsic)
                if len(closest_radar_data)>0:
                    closest_radar_data = np.concatenate((closest_radar_data, converted_radar_from_point), axis = 0)
                else:
                    closest_radar_data = converted_radar_from_point

                     
            # print("close radar data shape: ", closest_radar_data.shape)
            # print("close radar points: ", closest_radar_data)

            # =================== 1. Pass points without clustering. Consider cid=-1 and pop=1 for each point ===============
            radar_points = closest_radar_data[:, :5]    # (x, y, z, vx, vy)
            dummy_labels = -np.ones((closest_radar_data.shape[0], 1))
            dummy_cid = np.ones((closest_radar_data.shape[0], 1))

            radar_points = np.concatenate([radar_points, dummy_labels, dummy_cid], axis=1) 
            # print("Now radar data shape: ", radar_points.shape)

            self.radar_pcd.append(radar_points)

            # ================== 2. Alternative - cluster only core points and pass them ========================
      
            # core_points, core_labels, cluster_population = self.grid_dbscan_torch(closest_radar_data)

            # core_points_np = core_points.cpu().numpy() if hasattr (core_points, 'cpu') else core_points
            # core_labels_np = core_labels.cpu().numpy() if hasattr (core_labels, 'cpu') else core_labels


            # # #Build mapping from point to cluster info
            # # output = []
            # # point2cluster = {}
            # # for pt, cid in zip(core_points_np, core_labels_np):
            # #     key = tuple(np.round(pt[:3], 3))
            # #     point2cluster[key]=cid
            # # #print("point2cluster: ", point2cluster)
            
            # # for row, pix in zip(closest_radar_point_cam, points_img):
            # #     xyz = tuple(np.round(row[:3], 3))
            # #     print("xyz: ", xyz)
            # #     vx, vy = row[3], row[4]
            # #     if xyz in point2cluster:
            # #         print (f"{xyz} in point2cluster")
            # #         cid = int(point2cluster[xyz])
            # #         print("cid: ", cid)
            # #         pop = cluster_population.get(cid,0)
            # #         print("pop: ", pop)
            # #     else:
            # #         cid = -1
            # #         pop = 0
            # #     if abs(row[0])<=self.camX and row[2]<=self.camZ:
            # #         #u,v = float(pix[0]), float(pix[1])
            # #         output.append([row[0], row[1], row[2], vx, vy, cid, pop])
            # # #print("radar output: ", self.radar_pcd)
            # # self.radar_pcd.append(output)

            # #==============Logic 2: stack radar points===================#
            # output = []
            # point2cluster={}
            # for pt, cid in zip(core_points_np, core_labels_np):
            #     key = tuple(np.round(pt[:3], 3))
            #     #print("key: ", key)
            #     point2cluster[key]=cid
            
            # for row in closest_radar_data:
            #     xyz = tuple(np.round(row[:3], 3))
            #     #print("xyz: ", xyz)
            #     vx, vy = row[3], row[4]
            #     for k in point2cluster.keys():
            #         if all(abs(a - b) < 1e-6 for a, b in zip(k, xyz)):
            #             #print(f"{xyz} in point2cluster")
            #             cid = int(point2cluster[k])
            #             pop = cluster_population.get(cid,0)
            #             if abs(row[0])<=self.camX and row[2]<=self.camZ:
            #                 output.append([row[0], row[1], row[2], vx,vy, cid, pop])
            #                 #print("Appended: ", [row[0], row[1], row[2], vx,vy, cid, pop])
            # self.radar_pcd.append(output)

                


            #print("radar_pcd: ", self.radar_pcd.shape)
            #print("Radar points: ", radar_data)
            #print("Radar points shape: ", radar_data.shape)

            #Collect annotations
            annotations = []
            for ann_token in sample['anns']:
                ann = self.nusc.get('sample_annotation', ann_token)
                # Get category (semantic label)
                cat_name = ann['category_name']   # e.g. "vehicle.car"
                # 3D Bounding box
                center = np.array(ann['translation'])      # (x, y, z) global coords
                size = np.array(ann['size'])               # (w, l, h)
                quat = ann['rotation']                     # quaternion (w, x, y, z)
                # Extra: attribute ('moving', 'resting', etc), instance or track id
                attr = ann.get('attribute_name', None)
                instance_token = ann['instance_token']
                # ========================== Get image size from first load ==========================================
                if not hasattr(self, '_cached_img_size'):
                    temp_img = Image.open(self.image_paths[0])
                    self._cached_img_size = temp_img.size   # (w,h)
                    temp_img.close()
                
                width, height = self._cached_img_size

                # ========================================================================================

                center_cam, box_quat_cam, yaw_cam = self.transform_label_to_camera(center, quat, cam_ego_pose, cam_calib)
                cat_idx = self.category_to_idx(cat_name)
                #print("Center before transformation: ", center)
                #print("Center after transformation: ", center_cam)
                # Project center to image pixels -- returns shape (2,) [u, v]
                center_cam_hom = np.hstack([center_cam, 1])
                pixel = cam_intrinsic @ center_cam_hom[:3]
                #print("pixel: ", pixel)
                pixel = pixel[:2] / pixel[2] if pixel[2] != 0 else np.array([-1, -1]) # Homogeneous normalization
                #print("pixel after normalization: ", pixel)

                img_height, img_width = height, width
                #print("image size: ", (img_height, img_width))

                visible = (
                    center_cam[2] > 0 and
                    0 <= pixel[0] < img_width and
                    0 <= pixel[1] < img_height
                )
                if visible:
                    #print("Centers after visible: ", center_cam)
                    if center_cam[2]<self.camZ:
                        annotations.append({
                            'center': center_cam,       #Center of box in camera coordinate
                            'size': size,               #(width, length, height)
                            'yaw': yaw_cam,   #Quaternion in camera coordinate
                            'category_idx': cat_idx,
                            'pixel': pixel.astype(np.float32)
                        })
                #print("Annotations: ", annotations)
            self.labels.append(annotations)

            #i=i+1
            #if i>5:
            #    break
            #radar point format: x, y, z, vx, vy, rcs
            #Conduct DBSCAN clustering.
            #Get one latest point from each cluster with covariance (Mahalanobis distance)

            
        # self.images=np.array(self.images)
        self.radar_pcd = np.array(self.radar_pcd, dtype=object)
        self.cam_intrinsics = np.stack(self.cam_intrinsics, axis=0)     # (N, 3, 3)
        #print("image dataset shape: ", self.images.shape)
        #print("radar pcd dataset shape: ", self.radar_pcd.shape)
        #print("annotations shape: ", len(self.labels))
        #self.normalize()
        # self.images = self.images/255.0

    # ==================== CHANGE 6: Add helper method for lazy image loading ====================
    def _load_and_process_image(self, image_path):
        """Load and process a single image on-demand"""
        image = Image.open(image_path)
        image_resized = image.resize(self.image_size)
        image_resized = np.array(image_resized)
        image_resized = image_resized.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        image_resized = image_resized / 255.0  # Normalize
        return image_resized
    
    def _compute_segmentation(self, image_path):
        """Compute segmentation on-demand"""
        image = Image.open(image_path)
        image_seg = np.array(image)
        image_to_seg_resized = cv2.resize(image_seg, (self.seg_input_w, self.seg_input_h))
        image_to_seg = np.expand_dims(image_to_seg_resized, 0)
        seg_output = self.seg_model.predict(image_to_seg, verbose=0)  # (1, H, W, 19)
        seg_probs_19 = seg_output[0]
        seg_probs_19_resized = cv2.resize(seg_probs_19, self.image_size, interpolation=cv2.INTER_LINEAR)
        seg_probs_19_resized = seg_probs_19_resized / (seg_probs_19_resized.sum(axis=-1, keepdims=True) + 1e-8)
        seg_probs_12 = self.merge_segmentation_classes(seg_probs_19_resized, self.class_mapping, num_simplified_classes=12)
        return seg_probs_12.astype(np.float32)
    # ============================================================================================
    

    def extract_radar_data(self, radar_point):
        radar_path = os.path.join(self.nusc.dataroot, radar_point['filename'])
        radar_pc = RadarPointCloud.from_file(radar_path)
        x = radar_pc.points[0]
        y = radar_pc.points[1]
        z = radar_pc.points[2]
        vx = radar_pc.points[8]
        vy = radar_pc.points[9]
        rcs = radar_pc.points[6]
        radar_data = np.stack([x,y,z,vx,vy,rcs], axis = 1)
        return radar_data
        
    def grid_dbscan_torch(self, points):
        points = torch.from_numpy(points).float()
        #print("points to cluster: ", points)
        unit = self.eps/np.sqrt(3)


        loc_x = torch.floor((points[:, 0]+self.X)/unit).to(torch.int64)
        loc_y = torch.floor((-points[:,1]+self.Y)/unit).to(torch.int64)
        loc_z = torch.floor((points[:,2])/unit).to(torch.int64)

        # print("loc_x: ", loc_x)
        # print("loc_y: ", loc_y)
        # print("loc_z: ", loc_z)

        num_x = int(np.ceil(2*self.X / unit))
        num_y = int(np.ceil(2*self.Y / unit))
        num_z = int(np.ceil(self.Z / unit))

        grid_shape = num_x, num_y, num_z
        #print("Grid shape: ", grid_shape)
        # Mask for in-bound indices
        valid = (loc_x>=0) & (loc_x<num_x) & (loc_y>=0)& (loc_y<num_y) & (loc_z>=0) & (loc_z<num_z)
        loc_x, loc_y, loc_z = loc_x[valid], loc_y[valid], loc_z[valid]

        # print("loc_x after valid mask: ", loc_x)
        # print("loc_y after valid mask: ", loc_y)
        # print("loc_z after valid mask: ", loc_z)

        points = points[valid]

        # print("points after valid mask: ", points)

        #1D voxel indices
        voxel_idx = loc_x*(num_y*num_z) + loc_y*num_z + loc_z

        # print("voxel_idx: ", voxel_idx)

        #count points per voxel
        unique_voxels, counts = voxel_idx.unique(return_counts=True)
        # print("unique voxels: ", unique_voxels)
        # print("counts: ", counts)

        #Core voxel mask
        core_mask = counts >= self.min_pts
        core_voxels = unique_voxels[core_mask]
        # print("core_voxel_idx: ", core_voxels )

        #Build voxel index to grid (i,j,k) tuple mapping
        def unravel_index(idx, shape):
            k = idx % shape[2]
            j = ((idx-k)) // shape[2] % shape[1]
            i = ((idx-k-j*shape[2])) // (shape[1]*shape[2])

            return i,j,k
        
        cluster_labels = -torch.ones(unique_voxels.shape[0], dtype = torch.int64)

        #print("cluster labels: ", cluster_labels)
        cluster_id = 0
        # 5. BFS region growing on grid indices 
        visited = set()
        voxel_to_idx = {int(v): idx for idx, v in enumerate(unique_voxels)}

        for core_voxel in core_voxels:
            v_id = int(core_voxel)
            if v_id in visited:
                continue
            cluster_id += 1
            queue = deque([v_id])
            visited.add(v_id)
            # Assign cluster label
            cluster_labels[voxel_to_idx[v_id]] = cluster_id
            while queue:
                v = queue.popleft()
                i, j, k = unravel_index(v, grid_shape)
                # Iterate neighbors in 3D
                for di in [-1,0,1]:
                    for dj in [-1,0,1]:
                        for dk in [-1,0,1]:
                            if di==0 and dj==0 and dk==0: continue
                            ni, nj, nk = i+di, j+dj, k+dk
                            if 0<=ni<num_x and 0<=nj<num_y and 0<=nk<num_z:
                                nv = ni * (num_y * num_z) + nj * num_z + nk
                                if nv in voxel_to_idx and counts[voxel_to_idx[nv]] >= self.min_pts and nv not in visited:
                                    queue.append(nv)
                                    visited.add(nv)
                                    cluster_labels[voxel_to_idx[nv]] = cluster_id
        # 6. Assign labels back to points
        point_labels = -torch.ones(points.shape[0], dtype=torch.int64)
        for l_idx in range(unique_voxels.shape[0]):
            mask = (voxel_idx == unique_voxels[l_idx])
            if cluster_labels[l_idx] > 0:
                point_labels[mask] = cluster_labels[l_idx]
        # print("points: ", points)
        # print("point_labels: ", point_labels)

        cluster_pts_mask = point_labels>0
        core_points = points[cluster_pts_mask]
        core_labels = point_labels[cluster_pts_mask]
        # print("core points: ", core_points)
        # print("core_labels: ", core_labels)

        cluster_ids, cluster_counts = core_labels.unique(return_counts=True)
        cluster_population = {int(cid): int(cnt) for cid, cnt in zip(cluster_ids, cluster_counts)}
        # print("cluster counts: ",cluster_counts)
        # print("cluster population: ", cluster_population)
        #self.plot_clusters(core_points, core_labels)

        return core_points, core_labels, cluster_population
    
    def plot_clusters(self, points, labels):
        points = points.cpu().numpy() if hasattr(points, 'cpu') else points
        labels = labels.cpu().numpy() if hasattr(labels, 'cpu') else labels

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')

        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0,1,len(unique_labels)))

        for color, k in zip(colors, unique_labels):
            mask = (labels == k)
            if k == -1:  # noise points
                ax.scatter(points[mask,0], points[mask,1], points[mask,2], c='k', marker='.', s=10, alpha=0.25, label='Noise')
            else:
                ax.scatter(points[mask,0], points[mask,1], points[mask,2], c=color, marker='o', s=20, label=f'Cluster {k}')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('DBSCAN 3D Cluster Visualization')
        ax.legend()
        plt.show()
    
    def plot_bev_sample(self, idx=0, mode='train'):
        """
        BEV x–z plot (camera frame) of radar clusters + GT 3D boxes for a sample.
        x = lateral (camera right +), z = depth (forward +).
        """

        if mode == 'train':
            img = self.x_train_images[idx]          # not actually used, but available
            radar = np.array(self.x_train_radar[idx], dtype=float)  # (N,7): x,y,z,vx,vy,cid,pop
            labels = self.y_train[idx]
        else:
            img = self.x_val_images[idx]
            radar = np.array(self.x_val_radar[idx], dtype=float)
            labels = self.y_val[idx]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f'BEV x–z (camera frame) – sample {idx}')
        ax.set_xlabel('x (m) – lateral')
        ax.set_ylabel('z (m) – depth')
        ax.grid(True)
        ax.set_aspect('equal')

        # ---- Radar clusters in x–z ----
        if radar.shape[0] > 0:
            x = radar[:, 0] * self.camX        
            z = radar[:, 2] * self.camZ
            cid = radar[:, 5].astype(int)

            unique_cids = np.unique(cid)
            cmap = plt.cm.tab20
            for j, c in enumerate(unique_cids):
                mask = cid == c
                color = 'k' if c < 0 else cmap(j % 20)
                ax.scatter(x[mask], z[mask], s=10, c=[color],
                           alpha=0.7, label=f'cluster {c}' if c >= 0 else 'noise')

        # ---- GT boxes in x–z ----
        for ann in labels:
            # centers and sizes are normalized
            cx = ann['center'][0] * self.camX
            cz = ann['center'][2] * self.camZ
            w  = ann['size'][0] * 20.0   
            l  = ann['size'][1] * 40.0
            yaw = ann['yaw'] * np.pi     # your normalization: yaw / pi

            # box corners in local box frame (x–z plane)
            # x is lateral, z is depth; w along x, l along z
            corners_local = np.array([
                [-w/2, -l/2],
                [ w/2, -l/2],
                [ w/2,  l/2],
                [-w/2,  l/2],
                [-w/2, -l/2]
            ])

            R = np.array([[ np.cos(yaw), -np.sin(yaw)],
                          [ np.sin(yaw),  np.cos(yaw)]])
            corners_rot = (R @ corners_local.T).T
            corners_rot[:, 0] += cx
            corners_rot[:, 1] += cz

            ax.plot(corners_rot[:, 0], corners_rot[:, 1],
                    color='red', linewidth=1.5, alpha=0.9)

        ax.legend(loc='upper right', fontsize=8)
        plt.show()

    def plot_bev_xz_plane(self, idx=0, mode='train', x_range=(-50,50), z_range=(0,100)):
        """Plot radar points and GT bboxes"""
        fig, ax = plt.subplots(figsize=(12,10))

        x_min, x_max = x_range
        z_min, z_max = z_range

        if mode == 'train':
            img = self.x_train_images[idx]          # not actually used, but available
            radar = np.array(self.x_train_radar[idx], dtype=float)  # (N,7): x,y,z,vx,vy,cid,pop
            labels = self.y_train[idx]
        else:
            img = self.x_val_images[idx]
            radar = np.array(self.x_val_radar[idx], dtype=float)
            labels = self.y_val[idx]
        
        # ============= 1. Plot radar points =================================#
        if len(radar)>0:
            x_radar = radar[:, 0]
            z_radar = radar[:, 2]
            cluster_ids = radar[:, 5].astype(int)

            visible = (x_radar <= x_max) & (x_radar >= x_min) & \
                      (z_radar <= z_max) & (z_radar >= z_min)

            x_vis = x_radar[visible]
            z_vis = z_radar[visible]
            cid_vis = cluster_ids[visible]

            unique_cids = np.unique(cid_vis)
            cmap = plt.cm.tab20

            for cid in unique_cids:
                mask = cid_vis==cid
                if cid <0:
                    ax.scatter(x_vis[mask], z_vis[mask], c='red', marker='.', s=20, alpha=0.5, label='radar noise')
                else:
                    color = cmap((cid % 20) / 20.0)
                    ax.scatter(x_vis[mask], z_vis[mask], c=[color], marker='o', s=30, alpha=0.7, edgecolors='black', linewidth=0.5, label=f'radar cluster {cid}') 

        # ================2. Plot GT bounding boxes (green) =========================== #
        for ann in labels:
       
            cx = ann['center'][0]
            cz = ann['center'][2]
            w  = ann['size'][0]  
            l  = ann['size'][1]
            yaw = ann['yaw']    
            # box corners in local box frame (x–z plane)
            # x is lateral, z is depth; w along x, l along z
            corners_local = np.array([
                [-w/2, -l/2],
                [ w/2, -l/2],
                [ w/2,  l/2],
                [-w/2,  l/2],
                [-w/2, -l/2]
            ])

            R = np.array([[ np.cos(yaw), -np.sin(yaw)],
                          [ np.sin(yaw),  np.cos(yaw)]])
            corners_rot = (R @ corners_local.T).T
            corners_rot[:, 0] += cx
            corners_rot[:, 1] += cz

            # Plot box
            ax.plot(corners_rot[:, 0], corners_rot[:, 1],
                    color='green', linewidth=2, alpha=0.9, label='GT box')
            
            # GT center
            ax.scatter(cx, cz, c='green', marker='o', s=100, edgecolors='white', linewidths=2, label='GT center')

        ax.set_xlabel('x (m) - lateral (left ← | → right)', fontsize=12)
        ax.set_ylabel('z (m) - depth (forward →)', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', 'box')
        
        # Set limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)

        # Add origin indicator
        ax.axhline(0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
        ax.axvline(0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
        
        # Legend (remove duplicates)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                loc='upper right', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        plt.show()

        return fig, ax


    def transform_radar_to_camera(self,radar_points, radar_calib, radar_ego_pose,
                              cam_calib, cam_ego_pose, cam_intrinsic):
        """
        Transforms radar points (Nx3 for x,y,z, or Nx6 for x,y,z,vx,vy,rcs) to camera image plane.
        
        Args:
            radar_points: np.ndarray, shape (N, 3) or (N, 6)
            radar_calib: dict, radar sensor calibrated_sensor (has 'translation', 'rotation')
            radar_ego_pose: dict, radar ego pose ('translation', 'rotation')
            cam_calib: dict, camera calibrated_sensor (has 'translation', 'rotation')
            cam_ego_pose: dict, camera ego pose
            cam_intrinsic: np.ndarray, shape (3,3)
        Returns:
            points_img: np.ndarray, shape (N, 2), image pixel coordinates
            points_cam: np.ndarray, shape (N, 3), in camera coordinates
        """
        # Helper to build transform [R|t] as 4x4 matrix from translation + quaternion
        def get_transformation_matrix(translation, rotation):
            # rotation: [w, x, y, z] in nuScenes
            from pyquaternion import Quaternion
            q = Quaternion(rotation)
            R = q.rotation_matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = translation
            return T

        def apply_transform(points, T):
            N = points.shape[0]
            points_hom = np.hstack([points[:,:3], np.ones((N,1))])
            points_tf = (T @ points_hom.T).T
            return points_tf[:,:3]
        def apply_rotation(velocities, R):
            """Transform velocities (vx, vy, vz) using only rotation matrix"""
            return (R@velocities.T).T
        
        #Extract position and velocities
        positions = radar_points[:, :3] #(N,3) [x,y,z]
        velocities = radar_points[:, 3:5]   #(N,2) [vx, vy]
        velocities_3d = np.hstack([velocities, np.zeros((velocities.shape[0], 1))]) # (N,3), [vx,vy,vz]
        rcs = radar_points[:, 5:6]
        # Step 1: Radar sensor to ego
        radar2ego = get_transformation_matrix(radar_calib['translation'], radar_calib['rotation'])
        R_radar2ego = radar2ego[:3,:3]

        positions_ego = apply_transform(positions, radar2ego)
        velocities_ego = apply_rotation(velocities_3d, R_radar2ego)
        #points_ego = apply_transform(radar_points, radar2ego)
        # Step 2: Ego to global
        ego2global = get_transformation_matrix(radar_ego_pose['translation'], radar_ego_pose['rotation'])
        ego2global = get_transformation_matrix(radar_ego_pose['translation'], radar_ego_pose['rotation'])
        R_ego2global = ego2global[:3, :3]
        
        positions_global = apply_transform(positions_ego, ego2global)
        velocities_global = apply_rotation(velocities_ego, R_ego2global)
        #points_global = apply_transform(points_ego, ego2global)
        # Step 3: Global to ego at camera timestamp (inverse of camera ego pose)
        global2cam_ego = np.linalg.inv(get_transformation_matrix(cam_ego_pose['translation'], cam_ego_pose['rotation']))
        R_global2cam_ego = global2cam_ego[:3, :3]
    
        positions_cam_ego = apply_transform(positions_global, global2cam_ego)
        velocities_cam_ego = apply_rotation(velocities_global, R_global2cam_ego)
        #points_cam_ego = apply_transform(points_global, global2cam_ego)
        # Step 4: Ego to camera sensor (inverse of camera calibration)
        ego2cam = np.linalg.inv(get_transformation_matrix(cam_calib['translation'], cam_calib['rotation']))
        R_ego2cam = ego2cam[:3, :3]
    
        positions_in_cam = apply_transform(positions_cam_ego, ego2cam)
        velocities_in_cam = apply_rotation(velocities_cam_ego, R_ego2cam)
        #points_in_cam = apply_transform(points_cam_ego, ego2cam)
        # Step 5: Project 3D camera coords to image plane
        # Only use points in front of camera (z>0)
        front_mask = positions_in_cam[:,2] > 0
        positions_in_cam = positions_in_cam[front_mask]
        velocities_in_cam = velocities_in_cam[front_mask]
        rcs = rcs[front_mask]

        # Step 6: Project 3D camera coords to image plane
        points_img = (cam_intrinsic @ positions_in_cam.T).T
        points_img = points_img[:,:2] / points_img[:,2:3]  # project from homogeneous
        
        # Step 7: Combine transformed positions, velocities (vx, vy only), and RCS
        points_cam = np.hstack([
            positions_in_cam,      # (N, 3) [x, y, z]
            velocities_in_cam[:, :2],  # (N, 2) [vx, vy] - drop vz
            rcs                    # (N, 1) [rcs]
        ])  # Final shape: (N, 6)
        # points_in_cam = points_in_cam[front_mask]
        # points_img = (cam_intrinsic @ points_in_cam.T).T
        # points_img = points_img[:,:2] / points_img[:,2:3]  # project from homogeneous

        return points_img, points_cam

    
    def transform_label_to_camera(self, center_global, quat_global, cam_ego_pose, cam_calib):
        """
        Transform a box center and orientation from global coordinates to camera frame,
        using the same chain as the SimpleCameraRadarOverlay snippet.

        Returns:
            center_cam: (3,) in camera coordinates
            box_quat_cam: Quaternion in camera frame
            yaw_cam: float, rotation in BEV x–z plane (around camera y-axis)
        """

        # --- Build global -> camera ego (rot, trans) ---
        cam_ego_quat = Quaternion(cam_ego_pose['rotation'])
        global_to_cam_ego_rot = cam_ego_quat.rotation_matrix.T        # R_ge
        global_to_cam_ego_trans = np.array(cam_ego_pose['translation'])  # t_ge

        # --- Build camera ego -> camera (rot, trans) ---
        cam_calib_quat = Quaternion(cam_calib['rotation'])
        cam_ego_to_cam_rot = cam_calib_quat.rotation_matrix.T          # R_ec
        cam_ego_to_cam_trans = np.array(cam_calib['translation'])      # t_ec

        # ---------- CENTER ----------
        # Global -> camera ego
        center_cam_ego = global_to_cam_ego_rot @ (center_global - global_to_cam_ego_trans)
        # Camera ego -> camera
        center_cam = cam_ego_to_cam_rot @ (center_cam_ego - cam_ego_to_cam_trans)

        # ---------- ORIENTATION ----------
        box_quat_global = Quaternion(quat_global)

        # Global -> camera ego orientation
        # q_ego rotates ego->global, so global->ego is q_ego.inverse
        box_quat_ego = cam_ego_quat.inverse * box_quat_global

        # Get yaw angle (in cam_ego)
        yaw_cam = np.arctan2(2 * (box_quat_ego.w * box_quat_ego.z + box_quat_ego.x * box_quat_ego.y),
                 1 - 2 * (box_quat_ego.y**2 + box_quat_ego.z**2))

        # Camera ego -> camera orientation
        # q_calib rotates cam->ego, so ego->cam is q_calib.inverse
        box_quat_cam = cam_calib_quat.inverse * box_quat_ego

        # # Rotation matrix in camera frame
        # R_cam = box_quat_cam.rotation_matrix

        # # For camera coordinates x (right), y (down), z (forward),
        # # BEV is x–z plane; rotation in this plane is about y-axis.
        # # Use atan2 of the forward axis projected on x–z plane.
        # yaw_cam = np.arctan2(R_cam[0, 2], R_cam[2, 2])

        return center_cam, box_quat_cam, yaw_cam
    
    def create_cityscapes_to_simplified_mapping(self):
        """
        Create mapping from Cityscapes 19 classes to 12 simplified classes.
        Returns: numpy array of shape (19,) where each element is the target class ID
        """
        # Cityscapes class order (19 classes)
        cityscapes_classes = [
            'road',          # 0
            'sidewalk',      # 1
            'building',      # 2
            'wall',          # 3
            'fence',         # 4
            'pole',          # 5
            'traffic light', # 6
            'traffic sign',  # 7
            'vegetation',    # 8
            'terrain',       # 9
            'sky',           # 10
            'person',        # 11
            'rider',         # 12
            'car',           # 13
            'truck',         # 14
            'bus',           # 15
            'train',         # 16
            'motorcycle',    # 17
            'bicycle'        # 18
        ]
        
        # Mapping: cityscapes_idx -> simplified_idx
        mapping = np.array([
            0,   # road -> 0 (Road)
            1,   # sidewalk -> 1 (sidewalk)
            2,   # building -> 2 (Static.other)
            2,   # wall -> 2 (Static.other)
            2,   # fence -> 2 (Static.other)
            2,   # pole -> 2 (Static.other)
            2,   # traffic light -> 2 (Static.other)
            2,   # traffic sign -> 2 (Static.other)
            3,   # vegetation -> 3 (vegetation)
            4,   # terrain -> 4 (terrain)
            5,   # sky -> 5 (sky)
            6,   # person -> 6 (person)
            6,   # rider -> 6 (person) - rider merged with person
            7,   # car -> 7 (car)
            8,   # truck -> 8 (truck)
            9,   # bus -> 9 (bus)
            9,   # train -> 9 (bus) - train merged with bus
            10,  # motorcycle -> 10 (motorcycle)
            11   # bicycle -> 11 (bicycle)
        ], dtype=np.int32)
        
        return mapping

    def merge_segmentation_classes(self, seg_probs, mapping, num_simplified_classes=12):
        """
        Merge Cityscapes segmentation probabilities to simplified classes.
        
        Args:
            seg_probs: (H, W, 19) probability array from Cityscapes model
            mapping: (19,) array mapping cityscapes class -> simplified class
            num_simplified_classes: number of output classes (12)
        
        Returns:
            merged_probs: (H, W, 12) probability array with merged classes
        """
        H, W, C_orig = seg_probs.shape
        assert C_orig == 19, f"Expected 19 Cityscapes classes, got {C_orig}"
        
        # Initialize output
        merged_probs = np.zeros((H, W, num_simplified_classes), dtype=np.float32)
        
        # Accumulate probabilities for each simplified class
        for cityscapes_idx in range(19):
            simplified_idx = mapping[cityscapes_idx]
            merged_probs[:, :, simplified_idx] += seg_probs[:, :, cityscapes_idx]
        
        # Renormalize (should already sum to 1, but ensure numerical stability)
        merged_probs = merged_probs / (merged_probs.sum(axis=-1, keepdims=True) + 1e-8)
        
        return merged_probs

    
    def category_to_idx(self,annotation):
        if annotation == 'flat.drivable_surface':
            return 0
        elif annotation == 'flat.sidewalk':
            return 1
        elif annotation in ('static.manmade', 'static.other', 'movable_object','static_object'):
            return 2
        elif annotation == 'static.vegetation':
            return 3
        elif annotation == 'flat.terrain':
            return 4
        elif annotation == 'sky':
            return 5
        elif 'human' in annotation:
            return 6
        elif annotation == 'animal':
            return 7
        elif annotation in ('vehicle.car', 'vehicle.construction', 'vehicle.trailer'):
            return 
        elif annotation == 'vehicle.bus':
            return 9
        elif annotation == 'vehicle.motorcycle':
            return 10
        elif annotation == 'vehicle.bicycle':
            return 11
        else:
            return 2
    def train_val_split(self, val_ratio=0.2, random_seed=42):
        """Splits dataset paths and metadata (not loaded images)"""
        num_samples = len(self.image_paths)  # MODIFIED: Use image_paths length
        indices = np.arange(num_samples)
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, random_state=random_seed, shuffle=True
        )
        
        # ==================== CHANGE 7: Split paths, not loaded images ====================
        # Store indices instead of slicing image arrays
        self.train_indices = train_idx
        self.val_indices = val_idx
        
        # Split metadata (not images)
        self.x_train_camK = self.cam_intrinsics[train_idx]
        self.x_val_camK = self.cam_intrinsics[val_idx]
        self.x_train_radar = [self.radar_pcd[i] for i in train_idx]
        self.x_val_radar = [self.radar_pcd[i] for i in val_idx]
        self.y_train = [self.labels[i] for i in train_idx]
        self.y_val = [self.labels[i] for i in val_idx]


    def __len__(self):
        if self.mode == 'train':
            return len(self.train_indices)  
        elif self.mode == 'val':
            return len(self.val_indices)  
        else:
            return len(self.image_paths)  


    # ==================== CHANGE 8: Load images in __getitem__ ====================
    def __getitem__(self, index):
        if self.mode == 'train':
            actual_idx = self.train_indices[index]  # NEW: Map to original index
            image = self._load_and_process_image(self.image_paths[actual_idx])  # NEW: Load on-demand
            seg_mask = self._compute_segmentation(self.image_paths[actual_idx])  # NEW: Compute on-demand
            
            sample = {
                'images': image,  # MODIFIED: Loaded on-demand
                'radar': self.x_train_radar[index],
                'cam_K': self.x_train_camK[index],
                'labels': self.y_train[index],
                'seg_mask': seg_mask  # MODIFIED: Computed on-demand
            }
        elif self.mode == 'val':
            actual_idx = self.val_indices[index]  # NEW: Map to original index
            image = self._load_and_process_image(self.image_paths[actual_idx])  # NEW: Load on-demand
            seg_mask = self._compute_segmentation(self.image_paths[actual_idx])  # NEW: Compute on-demand
            
            sample = {
                'images': image,  # MODIFIED: Loaded on-demand
                'radar': self.x_val_radar[index],
                'cam_K': self.x_val_camK[index],
                'labels': self.y_val[index],
                'seg_mask': seg_mask  # MODIFIED: Computed on-demand
            }

        return sample



if __name__=="__main__":
    # ==================== INITIALIZATION ====================
    dataset = NuscDataset(
        (704, 256), 
        path='v1.0-trainval/v1.0-trainval_meta', 
        max_samples=10, 
        save_raw_img=False  
    )
    dataset.train_val_split()
    
    print(f"Total samples: {len(dataset.image_paths)}")
    print(f"Train samples: {len(dataset.train_indices)}")
    print(f"Val samples: {len(dataset.val_indices)}")
    
    # ==================== TEST LAZY LOADING ====================
    # Set mode to train
    dataset.mode = 'train'
    
    # Get sample using __getitem__ (lazy loading)
    test_idx = 1
    sample = dataset[test_idx] 
    
    # Extract data from sample dict
    train_image = sample['images']        # (C, H, W), normalized [0,1]
    train_radar = sample['radar']         # (N, 7): x,y,z,vx,vy,cid,pop
    train_labels = sample['labels']       # List of annotation dicts
    train_seg = sample['seg_mask']        # (H, W, 12) segmentation probs
    train_camK = sample['cam_K']          # (3, 3) camera intrinsic
    
    print(f"\n=== Loaded Sample {test_idx} ===")
    print(f"Image shape: {train_image.shape}")
    print(f"Radar points: {len(train_radar)}")
    print(f"Labels: {len(train_labels)}")
    print(f"Segmentation shape: {train_seg.shape}")
    print(f"Camera K shape: {train_camK.shape}")
    
    # ==================== PLOT RADAR CLUSTERS ====================
    pcd0 = np.array(train_radar)              # shape (N,7): x,y,z,vx,vy,cid,pop
    core_points0 = pcd0[:, :3]                # x,y,z
    core_labels0 = pcd0[:, 5].astype(int)     # cid
    
    dataset.plot_clusters(core_points0, core_labels0)
    
    # ==================== PLOT RGB IMAGE ====================
    # Convert from (C, H, W) to (H, W, C) for display
    train_image_display = train_image.transpose(1, 2, 0)  # (H, W, 3)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(train_image_display)
    plt.title(f'Train Sample {test_idx} - RGB Image')
    plt.axis('off')
    
    # ==================== PLOT SEGMENTATION ====================
    plt.subplot(1, 2, 2)
    seg_class_map = np.argmax(train_seg, axis=-1)  # (H, W)
    plt.imshow(seg_class_map, cmap='tab20')
    plt.title(f'Train Sample {test_idx} - Segmentation')
    plt.colorbar(label='Class ID')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # ==================== PLOT BEV (Modified Version) ====================
    # Create modified version that works with lazy loading
    def plot_bev_xz_plane_lazy(dataset, sample, labels, radar, 
                               x_range=(-50,50), z_range=(0,100)):
        """Plot radar points and GT bboxes - works with lazy loaded data"""
        fig, ax = plt.subplots(figsize=(12,10))
        
        x_min, x_max = x_range
        z_min, z_max = z_range
        
        # ============= 1. Plot radar points =================================
        if len(radar) > 0:
            x_radar = radar[:, 0]
            z_radar = radar[:, 2]
            cluster_ids = radar[:, 5].astype(int)
            
            visible = (x_radar <= x_max) & (x_radar >= x_min) & \
                      (z_radar <= z_max) & (z_radar >= z_min)
            
            x_vis = x_radar[visible]
            z_vis = z_radar[visible]
            cid_vis = cluster_ids[visible]
            
            unique_cids = np.unique(cid_vis)
            cmap = plt.cm.tab20
            
            for cid in unique_cids:
                mask = cid_vis == cid
                if cid < 0:
                    ax.scatter(x_vis[mask], z_vis[mask], c='red', marker='.', 
                             s=20, alpha=0.5, label='radar noise')
                else:
                    color = cmap((cid % 20) / 20.0)
                    ax.scatter(x_vis[mask], z_vis[mask], c=[color], marker='o', 
                             s=30, alpha=0.7, edgecolors='black', linewidth=0.5, 
                             label=f'radar cluster {cid}')
        
        # ================ 2. Plot GT bounding boxes =========================
        for ann in labels:
            cx = ann['center'][0]
            cz = ann['center'][2]
            w = ann['size'][0]
            l = ann['size'][1]
            yaw = ann['yaw']
            
            # Box corners in local frame (x–z plane)
            corners_local = np.array([
                [-w/2, -l/2],
                [ w/2, -l/2],
                [ w/2,  l/2],
                [-w/2,  l/2],
                [-w/2, -l/2]
            ])
            
            R = np.array([[ np.cos(yaw), -np.sin(yaw)],
                          [ np.sin(yaw),  np.cos(yaw)]])
            corners_rot = (R @ corners_local.T).T
            corners_rot[:, 0] += cx
            corners_rot[:, 1] += cz
            
            # Plot box
            ax.plot(corners_rot[:, 0], corners_rot[:, 1],
                    color='green', linewidth=2, alpha=0.9, label='GT box')
            
            # GT center
            ax.scatter(cx, cz, c='green', marker='o', s=100, 
                      edgecolors='white', linewidths=2, label='GT center')
        
        ax.set_xlabel('x (m) - lateral (left ← | → right)', fontsize=12)
        ax.set_ylabel('z (m) - depth (forward →)', fontsize=12)
        ax.set_title(f'BEV View - Sample (X-Z Plane)', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', 'box')
        
        # Set limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        
        # Add origin indicator
        ax.axhline(0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
        ax.axvline(0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
        
        # Legend (remove duplicates)
        handles, labels_legend = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_legend, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 loc='upper right', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    # Call the modified BEV plot
    plot_bev_xz_plane_lazy(dataset, sample, train_labels, train_radar)
    
    # ==================== TEST MULTIPLE SAMPLES ====================
    print("\n=== Testing Multiple Samples ===")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"Sample {i}: Image {sample['images'].shape}, "
              f"Radar {len(sample['radar'])} pts, "
              f"Labels {len(sample['labels'])} objs")
    
    # ==================== MEMORY CHECK ====================
    import psutil
    import os
    process = psutil.Process(os.getpid())
    print(f"\n=== Memory Usage ===")
    print(f"RAM: {process.memory_info().rss / 1e9:.2f} GB")





