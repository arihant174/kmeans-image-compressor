import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse

class KMeansImageCompressor:
    """
    A K-Means Image Compressor built from scratch using NumPy.
    This class clusters image pixels into K dominant colors to perform color quantization.
    """
    def __init__(self, k=16, max_iters=50, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def initialize_centroids(self, X):
        """Randomly pick k data points as initial centroids."""
        np.random.seed(42)  # For reproducibility
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[random_indices]

    def compute_distances(self, X):
        """
        Compute Euclidean distance from each point in X to each centroid efficiently.
        Using the mathematical expansion: ||x - c||^2 = ||x||^2 + ||c||^2 - 2(x . c^T)
        
        X shape: (N, 3)
        centroids shape: (k, 3)
        Output shape: (N, k)
        """
        # X_norm shape: (N, 1)
        X_norm = np.sum(X**2, axis=1, keepdims=True)
        # c_norm shape: (1, k)
        c_norm = np.sum(self.centroids**2, axis=1)
        # dot_product shape: (N, k)
        dot_product = np.dot(X, self.centroids.T)
        
        # Calculate squared distances
        distances_sq = X_norm + c_norm - 2 * dot_product
        
        # Add a small epsilon to avoid negative values due to floating point inaccuracies
        distances_sq = np.maximum(distances_sq, 0)
        return np.sqrt(distances_sq)

    def fit(self, X):
        """Run the K-Means algorithm and return the final cluster assignments."""
        self.centroids = self.initialize_centroids(X)
        
        for i in range(self.max_iters):
            # 1. Compute distances and assign points to the nearest centroid
            distances = self.compute_distances(X)
            labels = np.argmin(distances, axis=1)
            
            # 2. Update centroids to the mean of their assigned points
            new_centroids = np.zeros_like(self.centroids)
            for cluster_idx in range(self.k):
                # Get all points assigned to this cluster
                cluster_points = X[labels == cluster_idx]
                if len(cluster_points) > 0:
                    # New centroid is the mean of the points
                    new_centroids[cluster_idx] = cluster_points.mean(axis=0)
                else:
                    # If a cluster is empty, keep the old centroid
                    new_centroids[cluster_idx] = self.centroids[cluster_idx]
            
            # 3. Check for convergence (if centroids stop moving)
            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            print(f"Iteration {i+1}/{self.max_iters} - Centroid shift: {shift:.4f}")
            
            if shift < self.tol:
                print(f"Converged after {i+1} iterations!")
                break
                
        return labels

def compress_image(image_path, k=16):
    print(f"\n--- Loading image from {image_path} ---")
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Convert to NumPy array
    img_array = np.array(img)
    original_shape = img_array.shape  # (Height, Width, 3 channels)
    
    # Flatten the image into a 2D array: (Height * Width, 3)
    X = img_array.reshape(-1, 3)
    X = X.astype(np.float32)
    
    print(f"Original image size: {original_shape[0]}x{original_shape[1]} pixels.")
    print(f"Total pixels to process: {X.shape[0]:,}")
    print(f"\n--- Running K-Means clustering (K={k}) ---")
    
    # Run the K-Means algorithm
    compressor = KMeansImageCompressor(k=k, max_iters=50)
    labels = compressor.fit(X)
    
    # Reconstruct the image using the centroid colors
    print("\n--- Reconstructing compressed image ---")
    compressed_X = compressor.centroids[labels]
    
    # Ensure pixel values are integers between 0 and 255
    compressed_X = np.clip(compressed_X, 0, 255).astype(np.uint8)
    
    # Reshape back to the original image dimensions
    compressed_img_array = compressed_X.reshape(original_shape)
    
    # Save the compressed image
    dir_name = os.path.dirname(image_path)
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(dir_name, f"{name}_compressed_k{k}{ext}")
    
    compressed_img = Image.fromarray(compressed_img_array)
    compressed_img.save(output_path)
    print(f"Success! Compressed image saved to: {output_path}")

    # Plot original vs compressed side-by-side
    print("Generating visual comparison plot...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(img_array)
    axes[0].set_title(f"Original Image\n({X.shape[0]:,} unique pixels limit)")
    axes[0].axis('off')
    
    axes[1].imshow(compressed_img_array)
    axes[1].set_title(f"Compressed Image\n({k} Colors Only)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(dir_name, f"{name}_comparison_k{k}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Comparison plot saved to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Means Image Color Compressor")
    parser.add_argument("--image", type=str, required=True, help="Path to the image to compress")
    parser.add_argument("--k", type=int, default=16, help="Number of colors to compress to (default: 16)")
    
    args = parser.parse_args()
    compress_image(args.image, args.k)
