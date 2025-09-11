"""
Master Visualization Script for IDRiD DR-Vision Project
Generates all feature extraction visualizations for presentation
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def create_all_visualizations():
    """Generate all feature visualization images"""

    print("🎨 IDRiD DR-Vision: Feature Visualization Generator")
    print("=" * 60)

    # Set paths
    current_dir = Path(__file__).parent
    output_dir = current_dir / "output"
    sample_dir = current_dir / "sample_images"

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    # Find sample image
    sample_image_paths = [
        # Try sample images folder first
        sample_dir / "IDRiD_sample.jpg",
        # Try main dataset
        project_root / "data/raw/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_001.jpg",
        # Try alternative paths
        project_root / "data/raw/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_001.jpg",
    ]

    sample_image = None
    for path in sample_image_paths:
        if path.exists():
            sample_image = str(path)
            break

    if sample_image is None:
        print("❌ No sample retinal image found!")
        print("Please place a sample IDRiD image in visualization/sample_images/")
        return False

    print(f"📸 Using sample image: {Path(sample_image).name}")

    try:
        # Import visualization functions
        from hog_visualization import visualize_hog_features
        from graph_visualization import visualize_graph_structure
        from embeddings_visualization import visualize_gnn_embeddings
        from importance_visualization import visualize_xgboost_importance

        # Generate all visualizations
        print("\n1. 🔍 Generating HOG features visualization...")
        visualize_hog_features(sample_image, str(output_dir / "presentation_hog_features.png"))

        print("2. 🕸️ Generating Graph structure visualization...")
        visualize_graph_structure(sample_image, str(output_dir / "presentation_graph_structure.png"))

        print("3. 🧠 Generating GNN embeddings visualization...")
        visualize_gnn_embeddings(save_path=str(output_dir / "presentation_gnn_embeddings.png"))

        print("4. ⚡ Generating XGBoost feature importance...")
        visualize_xgboost_importance(str(output_dir / "presentation_xgboost_importance.png"))

        # Success message
        print("\n🎉 All visualizations created successfully!")
        print(f"📁 Output directory: {output_dir}")
        print("📄 Files generated:")
        for file in output_dir.glob("presentation_*.png"):
            print(f"   ✅ {file.name}")

        print("\n🎯 Ready for PowerPoint presentation!")
        return True

    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = create_all_visualizations()

