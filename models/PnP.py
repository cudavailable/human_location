import numpy as np
import cv2
import os
import time

class CameraCalibrator:
    """
    params:
    cam1_cali_path 和 cam2_cali_path：存储相机内参和畸变系数的文件路径。
    num_points_per_sample：每个样本点的数量，用于三角测量。
    correction_matrix_path：用于加载修正矩阵的路径，默认为 ./utils/correction_matrix.npy。
    """
    def __init__(self, cam1_cali_path, cam2_cali_path, num_points_per_sample, correction_matrix_path='./utils/correction_matrix.npy'):
        self.A_aug = np.load(correction_matrix_path)  # 记载修正矩阵
        self.K1, self.dist1 = self.init_cam(cam1_cali_path)  # 初始化两个相机的内参矩阵和畸变系数
        self.K2, self.dist2 = self.init_cam(cam2_cali_path)
        self.R1 = None  # 初始化旋转矩阵和位移向量为 None
        self.t1 = None
        self.R2 = None
        self.t2 = None
        self.num_points_per_sample = num_points_per_sample
        
    # 从指定文件中加载相机的内参矩阵和畸变系数，返回这两个值
    def init_cam(self, file_path):
        data = np.load(file_path)
        return data['mtx'], data['dist']
    
    # 计算相机的旋转矩阵和位移向量。如果成功，返回旋转矩阵和位移向量；否则返回 None
    def solve_pnp(self, camera_matrix, dist_coeffs, object_points, image_points):
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            object_points, image_points, camera_matrix, dist_coeffs
        )
        if success:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            print("Rotation Matrix:\n", rotation_matrix)
            print("Translation Vector:\n", translation_vector)
            return rotation_matrix, translation_vector
        else:
            print("PnP Failed")
            return None, None
    
    # 执行相机标定。计算两个相机的旋转矩阵和位移向量。将结果保存到指定的文件中
    def calibrate(self, imgpoints1, imgpoints2, objpoints, image_size, save_dir):
        
        self.R1, self.t1 = self.solve_pnp(self.K1, self.dist1, objpoints, imgpoints1)
        self.R2, self.t2 = self.solve_pnp(self.K2, self.dist2, objpoints, imgpoints2)

        save_path1 = os.path.join(save_dir, 'calibration_cam1.npz')
        save_path2 = os.path.join(save_dir, 'calibration_cam2.npz')
        np.savez(save_path1, mtx=self.K1, dist=self.dist1, rvecs=self.R1, tvecs=self.t1)
        np.savez(save_path2, mtx=self.K2, dist=self.dist2, rvecs=self.R2, tvecs=self.t2)

    # 根据相机的投影矩阵和图像点进行三角测量，得到世界坐标系下的三维点
    def triangulate_points(self, imgpoints1, imgpoints2):

        P1 = np.dot(self.K1, np.hstack((self.R1, self.t1)))
        P2 = np.dot(self.K2, np.hstack((self.R2, self.t2)))
        
        objpoints_h = cv2.triangulatePoints(P1, P2, imgpoints1.T, imgpoints2.T)
        objpoints_h = objpoints_h / objpoints_h[3]
        
        return objpoints_h[:3].T    # OpenCV中图像矩阵是列优先，而不是行优先
    
    # 计算绝对轨迹误差（ATE）
    def compute_ate_np(self, est_trajectory, gt_trajectory):
        delta_trans = gt_trajectory[0, :] - est_trajectory[0, :]
        adjusted_est_trajectory = est_trajectory + delta_trans
        error = np.sqrt(np.sum((gt_trajectory - adjusted_est_trajectory)**2, axis=1))
        ate = np.mean(error)
        return ate

    # 计算相对定位误差（RPE）
    def compute_rpe_np(self, est_trajectory, gt_trajectory):
        gt_deltas = gt_trajectory[1:, :] - gt_trajectory[:-1, :]
        est_deltas = est_trajectory[1:, :] - est_trajectory[:-1, :]
        error = np.sqrt(np.sum((gt_deltas - est_deltas)**2, axis=1))
        rpe = np.mean(error)
        return rpe

    # 计算不同类型的误差
    def calculate_errors(self, X_world_pred, X_world_target, imgpoints1, imgpoints2, threshes=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
        """
        X_world_pred: (N, 3) np.array
        X_world_target: (N, 3) np.array
        imgpoints1: (N, 2) np.array
        imgpoints2: (N, 2) np.array
        """
        errors = {'accuracy': {thresh: 0 for thresh in threshes}}
        
        X_world_pred_center = []
        X_world_target_center = []
        imgpoints1_center = []
        imgpoints2_center = []
        
        num_samples = int(X_world_pred.shape[0] / self.num_points_per_sample)
        for i in range(num_samples):
            start_idx = i * self.num_points_per_sample
            end_idx = start_idx + self.num_points_per_sample
            
            pred_sample = X_world_pred[start_idx:end_idx]
            target_sample = X_world_target[start_idx:end_idx]
            imgpoints1_sample = imgpoints1[start_idx:end_idx]
            imgpoints2_sample = imgpoints2[start_idx:end_idx]
            
            X_world_pred_center.append( np.mean(pred_sample, axis=0) )
            target_center = np.mean(target_sample, axis=0)
            
            center_3dx = target_center[0] / 1000
            center_3dy = target_center[1] / 1000
            center_3dz = target_center[2] / 1000
            
            X_world_center_BEV_aug = np.array([center_3dx, center_3dy, 1])
            X_world_center_BEV = X_world_center_BEV_aug.dot(self.A_aug).tolist()
            X_world_center = [ X_world_center_BEV[0] * 1000 , X_world_center_BEV[1] * 1000, center_3dz * 1000 ]
            
            X_world_target_center.append( np.array(X_world_center) )
            imgpoints1_center.append( np.mean(imgpoints1_sample, axis=0) )
            imgpoints2_center.append( np.mean(imgpoints2_sample, axis=0) )
        
        X_world_pred_center = np.array(X_world_pred_center).astype(np.float32)
        X_world_target_center = np.array(X_world_target_center).astype(np.float32)
        imgpoints1_center = np.array(imgpoints1_center).astype(np.float32)
        imgpoints2_center = np.array(imgpoints2_center).astype(np.float32)
        
        X_world_pred_center = X_world_pred_center / 1000.
        X_world_target_center = X_world_target_center / 1000.
        
        diff = X_world_pred_center - X_world_target_center
        distances = np.sqrt(np.sum(diff**2, axis=1))
        
        errors['error_X_mean'] = np.mean(distances)
        errors['error_X_median'] = np.median(distances)
        errors['error_X_std'] = np.std(distances)
        for thresh in threshes:
            acc = (distances <= thresh).astype(float).mean()
            errors['accuracy'][thresh] = acc
        
        projected1, _ = cv2.projectPoints(X_world_pred_center, self.R1, self.t1, self.K1, self.dist1)
        projected2, _ = cv2.projectPoints(X_world_pred_center, self.R2, self.t2, self.K2, self.dist2)
        
        reprojection_error1 = np.sqrt(np.sum((projected1.squeeze() - imgpoints1_center)**2, axis=1))
        reprojection_error2 = np.sqrt(np.sum((projected2.squeeze() - imgpoints1_center)**2, axis=1))

        errors['error_x1_mean'] = np.mean(reprojection_error1)
        errors['error_x1_median'] = np.median(reprojection_error1)
        errors['error_x1_std'] = np.std(reprojection_error1)

        errors['error_x2_mean'] = np.mean(reprojection_error2)
        errors['error_x2_median'] = np.median(reprojection_error2)
        errors['error_x2_std'] = np.std(reprojection_error2)
        
        errors['ATE_error'] = self.compute_ate_np(X_world_pred_center, X_world_target_center)
        errors['RPE_error'] = self.compute_rpe_np(X_world_pred_center, X_world_target_center)
        return errors
    
    # 逐样本进行三角测量，并记录时间
    def triangulate_points_samplewise(self, imgpoints1, imgpoints2):
        num_samples = imgpoints1.shape[0] // self.num_points_per_sample
        objpoints_pred = []
        elapsed_times = []

        for i in range(num_samples):
            start_idx = i * self.num_points_per_sample
            end_idx = start_idx + self.num_points_per_sample
            
            imgpoints1_sample = imgpoints1[start_idx:end_idx]
            imgpoints2_sample = imgpoints2[start_idx:end_idx]

            start_time = time.time()
            objpoints_pred_sample = self.triangulate_points(imgpoints1_sample, imgpoints2_sample)
            end_time = time.time()

            objpoints_pred.append(objpoints_pred_sample)
            elapsed_times.append(end_time - start_time)

        objpoints_pred = np.vstack(objpoints_pred)
        total_elapsed_time = np.sum(elapsed_times)
        return objpoints_pred, total_elapsed_time
    
    # 评估三角测量结果
    def evaluate(self, imgpoints1_test, imgpoints2_test, objpoints_target):
        objpoints_pred = self.triangulate_points(imgpoints1_test, imgpoints2_test)
        #_, total_elapsed_time = self.triangulate_points_samplewise(imgpoints1_test, imgpoints2_test)
        errors = self.calculate_errors(objpoints_pred, objpoints_target, imgpoints1_test, imgpoints2_test)
        return errors
