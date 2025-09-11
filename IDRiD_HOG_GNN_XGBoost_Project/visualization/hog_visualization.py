import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
import os
from pathlib import Path


def visualize_hog_features(image_path, save_path="hog_features_visualization.png"):
    """Generate HOG feature visualization for presentation"""

    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"File exists check: {os.path.exists(image_path)}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (1024, 1024))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)

    # Extract HOG features with visualization
    features, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        block_norm='L2-Hys'
    )

    # Create professional visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Retinal Image\n(4288×2848 → 1024×1024)', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Grayscale
    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title('Preprocessed Grayscale\nfor HOG Computation', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # HOG visualization
    axes[2].imshow(hog_image, cmap='hot')
    axes[2].set_title('HOG Feature Visualization\n9 Orientations, 324D per patch', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # Add technical details
    fig.suptitle('Histogram of Oriented Gradients (HOG) Feature Extraction',
                 fontsize=16, fontweight='bold', y=0.95)

    plt.figtext(0.5, 0.02,
                'HOG Parameters: 9 orientations • (8×8) pixels per cell • (2×2) cells per block • L2-Hys normalization',
                ha='center', fontsize=12, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.1)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ HOG visualization saved: {save_path}")


def find_sample_image():
    """Find available sample image in various locations"""

    current_dir = Path(__file__).parent
    project_root = current_dir.parent

    # Possible image locations
    possible_paths = [
        # In visualization/sample_images/
        current_dir / "sample_images" / "IDRiD_031.jpg",
        current_dir / "sample_images" / "IDRiD_001.jpg",

        # In main dataset
        project_root / "data/raw/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_031.jpg",
        project_root / "data/raw/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_001.jpg",

        # Alternative dataset structure
        project_root / "data/raw/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_031.jpg",
        project_root / "data/raw/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_001.jpg",
    ]

    print("🔍 Searching for sample images...")
    for path in possible_paths:
        if path.exists():
            print(f"✅ Found: {path}")
            return str(path)
        else:
            print(f"❌ Not found: {path}")

    return None


# Usage
if __name__ == "__main__":
    print("🎨 HOG Feature Visualization Generator")
    print("=" * 50)

    # Try to find a sample image automatically
    image_path = find_sample_image()

    if image_path:
        print(f"📸 Using image: {Path(image_path).name}")
        visualize_hog_features(image_path, "output/hog_features_presentation.png")
    else:
        print("❌ No sample image found!")
        print("\n💡 Solutions:")
        print("1. Copy an IDRiD image to visualization/sample_images/")
        print("2. Or specify the correct path manually:")
        print("   image_path = 'path/to/your/image.jpg'")
        print("   visualize_hog_features(image_path, 'output/hog_features_presentation.png')")
