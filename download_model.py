"""
Download MediaPipe Hand Landmarker Model
Downloads the required model file for hand detection
"""

import urllib.request
import os

def download_model():
    """Download the MediaPipe hand landmarker model"""
    
    model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    model_filename = "hand_landmarker.task"
    
    if os.path.exists(model_filename):
        print(f"✓ Model file '{model_filename}' already exists!")
        file_size = os.path.getsize(model_filename) / (1024 * 1024)  # Convert to MB
        print(f"  File size: {file_size:.2f} MB")
        return True
    
    print(f"Downloading MediaPipe hand landmarker model...")
    print(f"URL: {model_url}")
    print("This may take a moment...")
    
    try:
        # Download with progress
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.2f} MB / {mb_total:.2f} MB)", end='')
        
        urllib.request.urlretrieve(model_url, model_filename, progress_hook)
        print()  # New line after progress
        
        # Verify download
        if os.path.exists(model_filename):
            file_size = os.path.getsize(model_filename) / (1024 * 1024)
            print(f"✅ Download complete!")
            print(f"   File: {model_filename}")
            print(f"   Size: {file_size:.2f} MB")
            return True
        else:
            print("❌ Download failed - file not found after download")
            return False
            
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nManual download instructions:")
        print(f"1. Open this URL in your browser: {model_url}")
        print(f"2. Save the file as: {model_filename}")
        print("3. Place it in the same directory as your Python scripts")
        return False

if __name__ == "__main__":
    download_model()