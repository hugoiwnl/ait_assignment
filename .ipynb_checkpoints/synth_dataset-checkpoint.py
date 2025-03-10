import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import synthetic_shapes as ss
import cv2 as cv


def get_image(width, height):
    valid_shapes = ['checkerboard', 'ellipses', 'lines', 'multiple_polygons', 'star', 'polygon', 'cube', 'stripes']
    valid_postprocessing = ['none', 'blur', 'noise']
    post_processing_weights = [0.8, 0.1, 0.1]
    shape = np.random.choice(valid_shapes)
    post_processing = np.random.choice(valid_postprocessing, p=post_processing_weights)
    img = ss.generate_background(size=(width, height))
    
    shape_functions = {
        'checkerboard': ss.draw_checkerboard,
        'ellipses': ss.draw_ellipses,
        'lines': ss.draw_lines,
        'multiple_polygons': ss.draw_multiple_polygons,
        'star': ss.draw_star,
        'polygon': ss.draw_polygon,
        'cube': ss.draw_cube,
        'stripes': ss.draw_stripes
    }
    post_processing_functions = {
        'none': ss.do_nothing,
        'blur': ss.add_final_blur,
        'noise': ss.add_salt_and_pepper
    }
    
    points = shape_functions[shape](img)
    # print('post_processing', post_processing)
    img = post_processing_functions[post_processing](img)

    rgb_img = np.stack([img, img, img], axis=-1)  # Convert to RGB
    return rgb_img, points




class SyntheticDataset(Dataset):
    """Optimized synthetic dataset using PyTorch's native parallelism"""
    def __init__(self, num_samples, img_height, img_width, channels, seed=42):
        self.num_samples = num_samples
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.base_seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Unique seed per worker and index
            seed = self.base_seed + idx * worker_info.num_workers + worker_info.id
        else:
            seed = self.base_seed + idx
        
        np.random.seed(seed)
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                img, points = get_image(width=self.img_width, height=self.img_height)
                break  # Exit the loop if successful
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise  # Re-raise the exception if all attempts fail
        img = (img.astype(np.float32)) / 255.0  # Normalize in numpy

        # Vectorized label generation
        label = np.zeros((self.img_width, self.img_height), dtype=np.float32)
        if points is not None and len(points) > 0:
            points_array = np.array(points, dtype=np.int32)
            xs = points_array[:, 0]
            ys = points_array[:, 1]
            valid_mask = (xs >= 0) & (xs < self.img_height) & (ys >= 0) & (ys < self.img_width)
            valid_ys = ys[valid_mask]
            valid_xs = xs[valid_mask]
            label[valid_ys, valid_xs] = 1.0

        # Convert to tensors
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))  # (C, H, W)
        label_tensor = torch.from_numpy(label)                  # ( H, W)

        return img_tensor, label_tensor


def create_dataloader(dataset, batch_size=1, shuffle=False, num_workers=4, **kwargs):
    """Create optimized DataLoader with parallel workers"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Faster data transfer to GPU
        drop_last=True,   # Drop last batch if smaller than batch_size
        persistent_workers=num_workers > 0,  # Maintain workers between epochs
        **kwargs
    )


if __name__ == '__main__':
    # Benchmark parameters
    num_samples = 10000
    batch_size = 64
    img_height = 480
    img_width = 640
    channels = 3

    test_image, points = get_image(width=img_width, height=img_height)
    cv.imshow('img', test_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    
    # Test configurations
    configs = [
        {"name": "(4 workers)", "num_workers": 4},
        {"name": "(8 workers)", "num_workers": 8},
        {"name": "(12 workers)", "num_workers": 12},
        {"name": "(16 workers)", "num_workers": 16},
        

    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        
        dataset = SyntheticDataset(
            num_samples=num_samples,
            img_height=img_height,
            img_width=img_width,
            channels=channels,
            seed=42
        )
        
        loader = create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config["num_workers"]
        )
        
        start_time = time.time()
        for i, (images, labels) in enumerate(loader):
            if i % 5 == 0:
                print(f"Batch {i+1}/{num_samples//batch_size}")
        
        elapsed = time.time() - start_time
        throughput = num_samples / elapsed
        results.append({
            "name": config['name'],
            "elapsed": elapsed,
            "throughput": throughput
        })
        print(f"Time: {elapsed:.2f}s, Throughput: {throughput:.2f} samples/s")
    
    # Print results
    print("\nPerformance Summary:")
    print("=" * 60)
    print(f"{'Configuration':<25} {'Time (s)':<10} {'Throughput (samples/s)':<20}")
    print("-" * 60)
    for res in results:
        print(f"{res['name']:<25} {res['elapsed']:<10.2f} {res['throughput']:<20.2f}")
