import os
import json
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
import random
import math


class DatasetBuilder(Dataset):
    def __init__(self, dataPaths, train, train_people_ids, test_people_ids, noise=0, n_mask=0, camera_jitter=0, height=480, width=640, correction_matrix_path='./init/correction_matrix.npy'):
        self.A_aug = np.load(correction_matrix_path)
        self.dataPaths = dataPaths
        self.train = train
        self.noise = noise
        self.n_mask = n_mask
        self.camera_jitter = camera_jitter
        self.width = width
        self.height = height
        self.train_people_ids = train_people_ids
        self.test_people_ids = test_people_ids
        self.zhang_img0 = None
        self.zhang_img1 = None
        self.zhang_objs = None
        self.index_map = {
            5: 4,   # Left Shoulder
            6: 8,   # Right Shoulder
            7: 5,   # Left Elbow
            8: 9,   # Right Elbow
            9: 6,   # Left Wrist
            10: 10, # Right Wrist
            11: 12, # Left Hip
            12: 16, # Right Hip
            13: 13, # Left Knee
            14: 17, # Right Knee
            15: 14, # Left Ankle
            16: 18  # Right Ankle
        }
        self.num_keypoints = len(self.index_map)
        self.mmpose_skeleton_links = {
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (5, 11),
            (6, 12),
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16)
        }
        self.data_info = self.build_data()
        
    def __getitem__(self, index):
        x_img0, x_img0_center, x_img1, x_img1_center, X_world_center, X_world_center_BEV, UWB_BEV = self.data_info[index]
        return torch.Tensor(x_img0), torch.Tensor(x_img0_center), torch.Tensor(x_img1), torch.Tensor(x_img1_center), torch.Tensor(X_world_center), torch.Tensor(X_world_center_BEV), torch.Tensor(UWB_BEV)
        
    def __len__(self):
        return len(self.data_info)
    
    def align_joints_by_index(self, mmpose_cam0_joints, mmpose_cam1_joints, kinect_joints):
        aligned_mmpose0 = []
        aligned_mmpose1 = []
        aligned_kinect = []
        for mmp_idx, kin_idx in self.index_map.items():
            aligned_mmpose0.append(mmpose_cam0_joints[mmp_idx])
            aligned_mmpose1.append(mmpose_cam1_joints[mmp_idx])
            aligned_kinect.append(kinect_joints[kin_idx])
        return aligned_mmpose0, aligned_mmpose1, aligned_kinect
    
    def get_observations(self, keypoints, mask_indices):
        x_img = []
        for i, keypoint in enumerate(keypoints):
            u = (keypoint[0] / self.width) * 2 - 1
            v = (keypoint[1] / self.height) * 2 - 1
            if i in mask_indices:
                x_img.append(0.)
                x_img.append(0.)
                continue
            x_img.append(u)
            x_img.append(v)
        return x_img
    
    def get_KinectPoints(self, Kinect_joints):
        num_joints = len(Kinect_joints)
        sum_x = 0.
        sum_y = 0.
        sum_z = 0.
        for i, joint in enumerate(Kinect_joints):
            sum_x += joint[0] / 1000.
            sum_y += joint[2] / 1000.
            sum_z += -joint[1] / 1000.
        center_3dx = sum_x / num_joints
        center_3dy = sum_y / num_joints
        center_3dz = sum_z / num_joints
        
        X_world_center_BEV_aug = np.array([center_3dx, center_3dy, 1])
        X_world_center_BEV = X_world_center_BEV_aug.dot(self.A_aug).tolist()
        X_world_center = [ X_world_center_BEV[0], X_world_center_BEV[1], center_3dz ]
        
        return X_world_center, X_world_center_BEV

    def get_zhang(self, keypoints1, keypoints2, kinect_joints, mask_indices):
        for i, (keypoint1, keypoint2, joint) in enumerate(zip(keypoints1, keypoints2, kinect_joints)):
            if i in mask_indices:
                continue
            self.zhang_img0.append(keypoint1)
            self.zhang_img1.append(keypoint2)
            self.zhang_objs.append([ joint[0], joint[2], -joint[1] ])

    def add_mask(self, keypoints1, keypoints2, kinect_joints):
        num_points = len(keypoints1)
        mask_indices = np.random.choice(num_points, self.n_mask, replace=False)
        masked_keypoint1 = []
        masked_keypoint2 = []
        masked_joint = []
        for i in range(num_points):
            if i in mask_indices:
                masked_keypoint1.append([0, 0])
                masked_keypoint2.append([0, 0])
                masked_joint.append([0, 0, 0])                
            else:
                masked_keypoint1.append(keypoints1[i])
                masked_keypoint2.append(keypoints2[i])
                masked_joint.append(kinect_joints[i])
        return masked_keypoint1, masked_keypoint2, masked_joint, mask_indices

    def add_noise(self, keypoints_list):
        keypoints_array = np.array(keypoints_list, dtype=np.float32)
        noise = np.random.normal(0, self.noise, keypoints_array.shape)
        noisy_keypoints_array = keypoints_array + noise
        noisy_keypoints_list = noisy_keypoints_array.tolist()
        return noisy_keypoints_list

    def sample_keypoints_on_skeleton(self, keypoints, num_sample_times):
        sampled_keypoints = []
        links = list(self.mmpose_skeleton_links)
        for _ in range(num_sample_times):
            link = random.choice(links)
            point1, point2 = link
            direction_vec = [ 
                keypoints[point2][0] - keypoints[point1][0], 
                keypoints[point2][1] - keypoints[point1][1] 
            ]
            t = random.random()
            random_point = [ 
                keypoints[point1][0] + t * direction_vec[0],
                keypoints[point1][1] + t * direction_vec[1]
            ]
            sampled_keypoints.append(random_point)
        return sampled_keypoints
    
    def sample_mean_keypoints_on_skeleton(self, keypoints, num_points, num_sample_times = 30):
        sampled_mean_keypoints = []
        for _ in range(num_points):
            sampled_keypoints = self.sample_keypoints_on_skeleton(keypoints, num_sample_times)
            points_array = np.array(sampled_keypoints)
            mean_point = np.mean(points_array, axis=0).astype(np.float32)
            sampled_mean_keypoints.append( [ mean_point[0], mean_point[1] ] )
        return sampled_mean_keypoints
    
    
    def apply_camera_jitter(self, keypoints, translation_distance):
        direction_angle = random.uniform(0, 2 * math.pi)
        translation_vector = [
            translation_distance * math.cos(direction_angle),
            translation_distance * math.sin(direction_angle)
        ]
        jittered_keypoints = [
            [coord[0] + translation_vector[0], coord[1] + translation_vector[1]]
            for coord in keypoints
        ]
        return jittered_keypoints
    
    def get_img_center(self, keypoints):
        u_sum = 0.
        v_sum = 0.
        num_keypoints = len(keypoints)
        for i, keypoint in enumerate(keypoints):
            u = keypoint[0]
            v = keypoint[1]
            u = (keypoint[0] / self.width) * 2 - 1
            v = (keypoint[1] / self.height) * 2 - 1
            u_sum += u
            v_sum += v
        center_u = u_sum / num_keypoints
        center_v = v_sum / num_keypoints
        x_img_center = [ center_u, center_v, 1 ]
        return x_img_center
        
    def build_data(self):
        self.zhang_img0 = []
        self.zhang_img1 = []
        self.zhang_objs = []
        data = []
        dataPaths = sorted(self.dataPaths)
        dataPaths_split = []
        if self.train:
            train_people_ids = sorted(self.train_people_ids)
            dataPaths_split = [ dataPaths[idx] for idx in train_people_ids ]
        else:
            test_people_ids = sorted(self.test_people_ids)
            dataPaths_split = [ dataPaths[idx] for idx in test_people_ids ]

        for dataPath in dataPaths_split:
            with open(dataPath) as f:
                dataset_json = json.load(f)
            
            for item in dataset_json:
                img0_keypoints_zhang = item["MMPose-WebCam0"]
                img1_keypoints_zhang = item["MMPose-WebCam1"]
                Kinect_joints = item["Kinect-3dJoints"]
                UWB = item["UWB-2dPos"]
                mask_indices = []

                img0_keypoints_zhang, img1_keypoints_zhang, Kinect_joints = self.align_joints_by_index(img0_keypoints_zhang, img1_keypoints_zhang, Kinect_joints)
                X_world_center, X_world_center_BEV = self.get_KinectPoints(Kinect_joints)
                UWB.append(X_world_center[2])
                
                x_img0_center = self.get_img_center(img0_keypoints_zhang)
                x_img1_center = self.get_img_center(img1_keypoints_zhang)
                
                if self.camera_jitter > 0:
                    img0_keypoints_zhang = self.apply_camera_jitter(img0_keypoints_zhang, self.camera_jitter)
                    img1_keypoints_zhang = self.apply_camera_jitter(img1_keypoints_zhang, self.camera_jitter)
                if self.noise > 0:
                    img0_keypoints_zhang = self.add_noise(img0_keypoints_zhang)
                    img1_keypoints_zhang = self.add_noise(img1_keypoints_zhang)
                if self.n_mask > 0:
                    mask_indices = np.random.choice(len(img0_keypoints_zhang), self.n_mask, replace=False)
                self.get_zhang(img0_keypoints_zhang, img1_keypoints_zhang, Kinect_joints, mask_indices)

                
                img0_keypoints = item["MMPose-WebCam0"]
                img1_keypoints = item["MMPose-WebCam1"]
                Kinect_joints = item["Kinect-3dJoints"]

                img0_keypoints = self.sample_mean_keypoints_on_skeleton(img0_keypoints, self.num_keypoints)
                img1_keypoints = self.sample_mean_keypoints_on_skeleton(img1_keypoints, self.num_keypoints)

                if self.camera_jitter > 0:
                    img0_keypoints = self.apply_camera_jitter(img0_keypoints, self.camera_jitter)
                    img1_keypoints = self.apply_camera_jitter(img1_keypoints, self.camera_jitter)
                if self.noise > 0:
                    img0_keypoints = self.add_noise(img0_keypoints)
                    img1_keypoints = self.add_noise(img1_keypoints)

                x_img0 = self.get_observations(img0_keypoints, mask_indices)
                x_img1 = self.get_observations(img1_keypoints, mask_indices)
                data.append( ( x_img0, x_img0_center, x_img1, x_img1_center, X_world_center, X_world_center_BEV, UWB ) )
        
        self.zhang_img0 = np.array(self.zhang_img0).astype(np.float32)
        self.zhang_img1 = np.array(self.zhang_img1).astype(np.float32)
        self.zhang_objs = np.array(self.zhang_objs).astype(np.float32)
        return data