import cv2
import numpy as np
import sys
import os
from tqdm import tqdm

def fast(img):
    detector = cv2.FastFeatureDetector_create()
    detector.setThreshold(10)  # Set threshold
    corners = detector.detect(img, None)
    
    detections = np.zeros(img.shape[:2], dtype=float)  # Ensure it's 2D
    for c in corners:
        x, y = map(int, c.pt)  # Convert to integer indices
        detections[y, x] = c.response  # Assign response value
    return detections

if __name__ == '__main__':
    # check the number of arguments
    if len(sys.argv) != 2:
        print("Usage: python export_detections_repeatability.py <mode>")
        sys.exit(1)
    # check the argument for the mode (viewpoint change, illumination change, both)
    if sys.argv[1] not in ['viewpoint', 'illumination', 'both']:
        print("Invalid mode. Choose one of 'viewpoint', 'illumination', 'both'.")
        sys.exit(1)
    else:
        mode = sys.argv[1]
    
    experiment_name = 'FAST_experiment_'+mode
    results_path = '/results'
    data_path = '/evaluation/HPatches'
    
    # Load the HPatches dataset, i.e. all directories starting with v_ in data_path
    if mode == 'viewpoint':
        seqs = [f for f in os.listdir(data_path) if f.startswith('v_')]
    elif mode == 'illumination':
        seqs = [f for f in os.listdir(data_path) if f.startswith('i_')]
    elif mode == 'both':
        seqs = [f for f in os.listdir(data_path)]
    
    for seq in tqdm(seqs, 'sequences'):
        # load images, i.e. files terminating in .ppm
        files = [f for f in os.listdir(os.path.join(data_path, seq)) if f.endswith('.ppm')]
        files = sorted(files)
        imgs = [cv2.imread(os.path.join(data_path, seq, f), cv2.IMREAD_GRAYSCALE) for f in files]
        # get homographies
        h_files = [f for f in os.listdir(os.path.join(data_path, seq)) if f.startswith('H_')]
        # Compute detections
        detections = [fast(img) for img in imgs]
        # Save detections
        seq_path = os.path.join(results_path, experiment_name, seq)
        os.makedirs(seq_path, exist_ok=True)
        # compy over the homography files
        for h_file in h_files:
            os.system(f'cp {os.path.join(data_path, seq, h_file)} {os.path.join(seq_path, h_file)}')
        for i, det in enumerate(detections):
            np.save(os.path.join(seq_path, f'{i+1:01d}_det.npy'), det)
    print('Detections exported.')

    

    