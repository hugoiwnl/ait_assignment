import numpy as np
import os
from tqdm import tqdm

def compute_repeatability_HPatches(path_to_results, num_kpts, dist_thresh):
    """
    Computes the average repeatability on the HPatches dataset.

    Args:
        path_to_results: Path to the folder containing the results.  See the
                         problem description for the expected folder structure.
        num_kpts: The number of keypoints to keep (top-k).
        dist_thresh: The distance threshold (in pixels) for repeatability.

    Returns:
        A tuple containing:
        - average_repeatability_illumination:  Average repeatability for illumination sequences (or 0.0 if no illumination sequences).
        - average_repeatability_viewpoint: Average repeatability for viewpoint sequences (or 0.0 if no viewpoint sequences).
    """

    def compute_repeatability_single_pair(prob1, prob2, H, num_kpts, dist_thresh):
        """
        Compute the repeatability for a single pair of images.

        Args:
            prob1: Probability map for the first image (original).  A 2D numpy array.
            prob2: Probability map for the second image (warped). A 2D numpy array.
            H: The 3x3 homography matrix mapping points from image1 to image2.
            keep_k_points: The maximum number of keypoints to consider (top-k).
            distance_thresh: The distance threshold (in pixels) for considering a match.

        Returns:
            The repeatability score (float).
        """

        def warp_keypoints(keypoints, H):
            num_points = keypoints.shape[0]
            homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))], axis=1)
            warped_points = np.dot(homogeneous_points, np.transpose(H))
            return warped_points[:, :2] / warped_points[:, 2:]

        def filter_keypoints(points, shape):
            mask = (points[:, 0] >= 0) & (points[:, 0] < shape[0]) & \
                   (points[:, 1] >= 0) & (points[:, 1] < shape[1])
            return points[mask, :]

        def keep_true_keypoints(points, H, shape):
            warped_points = warp_keypoints(points[:, [1, 0]], H)
            warped_points[:, [0, 1]] = warped_points[:, [1, 0]]
            mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[0]) & \
                   (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[1])
            return points[mask, :]

        def select_k_best(points, k):
            sorted_prob = points[points[:, 2].argsort(), :2]
            start = min(k, points.shape[0])
            return sorted_prob[-start:, :]

        shape = prob2.shape  # Use prob2 (warped image) shape for filtering

        # --- Extract and filter keypoints from prob1 (original image) ---
        keypoints1_indices = np.where(prob1 > 0)
        prob1_values = prob1[keypoints1_indices[0], keypoints1_indices[1]]
        keypoints1 = np.stack([keypoints1_indices[0], keypoints1_indices[1], prob1_values], axis=-1)

        # --- Extract and filter keypoints from prob2 (warped image) ---
        keypoints2_indices = np.where(prob2 > 0)
        prob2_values = prob2[keypoints2_indices[0], keypoints2_indices[1]]
        keypoints2 = np.stack([keypoints2_indices[0], keypoints2_indices[1], prob2_values], axis=-1)
        keypoints2 = keep_true_keypoints(keypoints2, np.linalg.inv(H), prob1.shape)  # Filter based on inverse H

        # --- Warp keypoints1 to the coordinate frame of image2 ---
        true_warped_keypoints1 = warp_keypoints(keypoints1[:, [1, 0]], H)  # Swap to (x, y)
        true_warped_keypoints1 = np.stack([true_warped_keypoints1[:, 1], true_warped_keypoints1[:, 0], keypoints1[:, 2]], axis=-1) # Add back the prob
        true_warped_keypoints1 = filter_keypoints(true_warped_keypoints1, shape)  # Filter based on prob2 shape


        # --- Select top-k keypoints ---
        keypoints2 = select_k_best(keypoints2, num_kpts)
        true_warped_keypoints1 = select_k_best(true_warped_keypoints1, num_kpts)

        # --- Compute Repeatability ---
        N1 = true_warped_keypoints1.shape[0]
        N2 = keypoints2.shape[0]

        if N1 == 0 or N2 == 0:
            return 0.0  # Handle cases with no keypoints

        true_warped_keypoints1 = np.expand_dims(true_warped_keypoints1, 1)
        keypoints2 = np.expand_dims(keypoints2, 0)
        norm = np.linalg.norm(true_warped_keypoints1 - keypoints2, ord=None, axis=2)

        count1 = 0
        count2 = 0
        if N2 != 0:
            min1 = np.min(norm, axis=1)
            count1 = np.sum(min1 <= dist_thresh)
        if N1 != 0:
            min2 = np.min(norm, axis=0)
            count2 = np.sum(min2 <= dist_thresh)

        repeatability = (count1 + count2) / (N1 + N2)
        return repeatability


    illumination_repeatabilities = []
    viewpoint_repeatabilities = []

    for sequence_name in tqdm(os.listdir(path_to_results)):
        sequence_path = os.path.join(path_to_results, sequence_name)
        if not os.path.isdir(sequence_path):
            continue

        if sequence_name.startswith('i_'):
            repeatabilities = illumination_repeatabilities
        elif sequence_name.startswith('v_'):
            repeatabilities = viewpoint_repeatabilities
        else:
            continue  # Skip sequences that don't follow the naming convention

        # Load the first detection map
        prob1 = np.load(os.path.join(sequence_path, "1_det.npy"))

        # Iterate through the other images in the sequence
        for i in range(2, 7):
            prob2_path = os.path.join(sequence_path, f"{i}_det.npy")
            H_path = os.path.join(sequence_path, f"H_1_{i}")

            if not os.path.exists(prob2_path) or not os.path.exists(H_path):
                print(f"Warning: Missing files for sequence {sequence_name}, image {i}. Skipping.")
                continue

            prob2 = np.load(prob2_path)
            H = np.loadtxt(H_path)  # Load homography as text file

            repeatability = compute_repeatability_single_pair(prob1, prob2, H, num_kpts, dist_thresh)
            repeatabilities.append(repeatability)

    average_repeatability_illumination = np.mean(illumination_repeatabilities) if illumination_repeatabilities else 0.0
    average_repeatability_viewpoint = np.mean(viewpoint_repeatabilities) if viewpoint_repeatabilities else 0.0

    return average_repeatability_illumination, average_repeatability_viewpoint