# K-Means Image Color Compressor

This project implements the K-Means clustering algorithm completely from scratch using **Python and NumPy**, without using `scikit-learn`. 

It performs **Color Quantization**, a lossy compression technique that reduces the number of distinct colors used in an image (e.g., from 16 million to 16), which is the foundational concept behind indexed formats like PNG-8 and GIF.

## How It Works
1. **Pixels to 3D Points**: Every pixel is treated as a 3D point (Red, Green, Blue) in space.
2. **K-Means Clustering**: The algorithm groups all pixels into `K` clusters based on color similarity (Euclidean distance).
3. **Reconstruction**: Every pixel's color is replaced by the average color (centroid) of its assigned cluster.

## Setup & Running

### Prerequisites
Make sure you have `numpy`, `matplotlib`, and `pillow` installed:
```bash
pip install numpy matplotlib pillow
```

### Quick Start
1. **Download a sample image:**
   ```bash
   python download_sample.py
   ```
2. **Run the Compressor:**
   ```bash
   python compressor.py --image sample_image.jpg --k 16
   ```

You will see:
- Iteration progress (until the cluster centers stop moving).
- A new compressed image saved in your folder.
- A side-by-side visual comparison PNG.


