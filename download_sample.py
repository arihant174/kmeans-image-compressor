import urllib.request
import os

def download_sample_image():
    # URL of a colorful sample image (Unsplash)
    url = "https://images.unsplash.com/photo-1493246507139-91e8fad9978e?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80"
    output_path = "sample_image.jpg"
    
    if not os.path.exists(output_path):
        print("Downloading sample image...")
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"Success! Downloaded {output_path}")
            print("\nYou can now run the compressor with:")
            print("python compressor.py --image sample_image.jpg --k 16")
        except Exception as e:
            print(f"Failed to download image: {e}")
    else:
        print(f"Sample image '{output_path}' already exists.")
        print("\nYou can run the compressor with:")
        print("python compressor.py --image sample_image.jpg --k 16")

if __name__ == "__main__":
    download_sample_image()
