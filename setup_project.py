"""
Project Structure Setup Script
Creates the necessary directory structure for the Gesture-Controlled Smart Camera project
"""

import os

# Define project structure
project_dirs = [
    "src",                          # Main source code
    "src/models",                   # Model architectures
    "src/utils",                    # Utility functions
    "data",                         # Data storage
    "data/raw",                     # Raw collected images
    "data/processed",               # Processed landmark data
    "data/train",                   # Training dataset
    "data/test",                    # Test dataset
    "models",                       # Saved trained models
    "outputs",                      # Captured photos
    "notebooks",                    # Jupyter notebooks for experiments
    "logs",                         # Training logs
]

def create_project_structure():
    """Create all necessary directories"""
    for directory in project_dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}/")
    
    # Create __init__.py files for Python packages
    init_files = ["src/__init__.py", "src/models/__init__.py", "src/utils/__init__.py"]
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write("# Auto-generated __init__.py\n")
        print(f"✓ Created: {init_file}")
    
    print("\n✅ Project structure created successfully!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run webcam test: python test_webcam.py")
    print("3. Run MediaPipe test: python test_mediapipe.py")

if __name__ == "__main__":
    create_project_structure()