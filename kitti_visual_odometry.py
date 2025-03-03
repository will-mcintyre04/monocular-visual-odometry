import os
import numpy as np
import cv2
from matplotlib import pyplot as plt 

from lib.visualization import plotting
from lib.visualization.video import play_trip

from tqdm import tqdm

class VisualOdometry():
    def __init__(self, data_dir):
        # Initialize camera calibration, ground truth poses, and image data
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'))
        self.images = self._load_images(os.path.join(data_dir, 'image_l'))
        self.orb = cv2.ORB_create(3000)  # ORB feature detector with 3000 keypoints
        
        # FLANN-based matcher with LSH index
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    @staticmethod
    def _load_calib(filepath):
        """
        Loads the camera calibration parameters.
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]  # Extract intrinsic parameters
        return K, P

    @staticmethod
    def _load_poses(filepath):
        """
        Loads ground truth poses.
        """
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ').reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))  # Convert to homogeneous transformation matrix
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        """
        Loads grayscale images from directory.
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _form_transf(R, t):
        """
        Forms transformation matrix from rotation matrix and translation vector.
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
        """
        Detects keypoints and computes descriptors for consecutive frames.
        """
        keypoints1, descriptors1 = self.orb.detectAndCompute(self.images[i - 1], None)
        keypoints2, descriptors2 = self.orb.detectAndCompute(self.images[i], None)
        matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)
        
        # Lowe's ratio test to filter good matches
        good = [m for m, n in matches if m.distance < 0.5 * n.distance]
        
        q1 = np.float32([keypoints1[m.queryIdx].pt for m in good])
        q2 = np.float32([keypoints2[m.trainIdx].pt for m in good])

        # Visualization of matches
        draw_params = dict(matchColor=-1, singlePointColor=None, matchesMask=None, flags=2)
        img3 = cv2.drawMatches(self.images[i], keypoints1, self.images[i - 1], keypoints2, good, None, **draw_params)
        cv2.imshow("image", img3)
        cv2.waitKey(750)

        return q1, q2

    def get_pose(self, q1, q2):
        """
        Estimates camera pose using Essential matrix decomposition.
        """
        Essential, mask = cv2.findEssentialMat(q1, q2, self.K)
        R, t = self.decomp_essential_mat(Essential, q1, q2)
        return self._form_transf(R, t)

    def decomp_essential_mat(self, E, q1, q2):
        """
        Decomposes the Essential matrix to extract the correct rotation and translation pair.
        """
        R1, R2, t = cv2.decomposeEssentialMat(E)
        transformations = [self._form_transf(R1, t.flatten()), self._form_transf(R2, t.flatten()),
                           self._form_transf(R1, -t.flatten()), self._form_transf(R2, -t.flatten())]

        K = np.concatenate((self.K, np.zeros((3, 1))), axis=1)
        projections = [K @ T for T in transformations]
        positives = []

        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]
            total_sum = np.sum(Q2[2, :] > 0) + np.sum(Q1[2, :] > 0)
            positives.append(total_sum)

        max_index = np.argmax(positives)
        return transformations[max_index][:3, :3], transformations[max_index][:3, 3]


def main():
    data_dir = 'KITTI_sequence_1'  # Path to dataset directory
    vo = VisualOdometry(data_dir)

    play_trip(vo.images)  # Visualize image sequence
    
    gt_path = []  # Ground truth path
    estimated_path = []  # Estimated path

    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
        if i == 0:
            cur_pose = gt_pose
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
        
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

    # Visualize the estimated and ground truth paths
    plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + ".html")

if __name__ == "__main__":
    main()