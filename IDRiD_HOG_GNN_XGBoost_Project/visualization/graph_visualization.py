import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import os

# Import NetworkX with error handling
try:
    import networkx as nx

    print("✅ NetworkX imported successfully")
except ImportError as e:
    print(f"❌ NetworkX import failed: {e}")
    print("💡 Install with: pip install networkx")
    nx = None

try:
    from sklearn.neighbors import NearestNeighbors

    print("✅ scikit-learn imported successfully")
except ImportError as e:
    print(f"❌ scikit-learn import failed: {e}")
    print("💡 Install with: pip install scikit-learn")
    NearestNeighbors = None


def visualize_graph_structure(image_path, save_path="graph_structure_visualization.png"):
    """Create spatial graph visualization for presentation"""

    # Check if required libraries are available
    if nx is None:
        print("❌ NetworkX not available. Please install: pip install networkx")
        return

    if NearestNeighbors is None:
        print("❌ scikit-learn not available. Please install: pip install scikit-learn")
        return

    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"File exists check: {os.path.exists(image_path)}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (1024, 1024))

    # Generate patch coordinates (32x32 patches with stride 32)
    patch_size = 32
    stride = 32
    h, w = 1024, 1024

    coordinates = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            center_x = x + patch_size // 2
            center_y = y + patch_size // 2
            coordinates.append([center_x, center_y])

    coordinates = np.array(coordinates)
    print(f"📊 Generated {len(coordinates)} patch coordinates")

    # Build k-NN graph (k=8)
    k = 8
    try:
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(coordinates)
        distances, indices = nbrs.kneighbors(coordinates)
        print("✅ k-NN computation successful")
    except Exception as e:
        print(f"❌ k-NN computation failed: {e}")
        return

    # Create NetworkX graph
    try:
        G = nx.Graph()  # This line was causing the error
        for i in range(len(coordinates)):
            G.add_node(i, pos=coordinates[i])
            for j in range(1, k + 1):  # Skip self (index 0)
                neighbor_idx = indices[i][j]
                G.add_edge(i, neighbor_idx)

        print(f"🕸️ Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    except Exception as e:
        print(f"❌ Graph creation failed: {e}")
        return

    # Create visualization
    try:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))

        # Original image with patch grid
        axes[0].imshow(image_resized)

        # Draw patch grid
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                rect = patches.Rectangle((x, y), patch_size, patch_size,
                                         linewidth=0.5, edgecolor='yellow', facecolor='none', alpha=0.7)
                axes[0].add_patch(rect)

        axes[0].set_title('Retinal Image with Patch Grid\n32×32 patches, stride=32',
                          fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Image with nodes
        axes[1].imshow(image_resized)
        axes[1].scatter(coordinates[:, 0], coordinates[:, 1], c='red', s=30, alpha=0.8)
        axes[1].set_title(f'Patch Centers as Graph Nodes\n{len(coordinates)} nodes per image',
                          fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # Graph structure with edges (sample subset for clarity)
        axes[2].imshow(image_resized, alpha=0.3)  # Faded background

        # Draw sample of edges (every 10th node to avoid clutter)
        sample_nodes = list(range(0, len(coordinates), 10))
        edges_drawn = 0

        for node in sample_nodes:
            x, y = coordinates[node]
            axes[2].scatter(x, y, c='red', s=50, alpha=0.9)

            # Draw edges to neighbors
            for neighbor in G.neighbors(node):
                if neighbor in sample_nodes:
                    nx_coord, ny_coord = coordinates[neighbor]  # Renamed to avoid confusion
                    axes[2].plot([x, nx_coord], [y, ny_coord], 'b-', alpha=0.6, linewidth=1)
                    edges_drawn += 1

        axes[2].set_title(f'k-NN Spatial Graph Structure\nk=8 neighbors per node\n({edges_drawn} sample edges shown)',
                          fontsize=14, fontweight='bold')
        axes[2].axis('off')

        # Add overall title and info
        fig.suptitle('Spatial Graph Construction for GNN Processing',
                     fontsize=16, fontweight='bold', y=0.95)

        plt.figtext(0.5, 0.02,
                    f'Graph Properties: {len(coordinates)} nodes • {G.number_of_edges()} edges • k-NN connectivity preserves spatial relationships',
                    ha='center', fontsize=12, style='italic')

        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.1)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Graph visualization saved: {save_path}")

    except Exception as e:
        print(f"❌ Visualization creation failed: {e}")
        import traceback
        traceback.print_exc()


def find_sample_image():
    """Find available sample image in various locations"""

    current_dir = Path(__file__).parent
    project_root = current_dir.parent

    # Possible image locations
    possible_paths = [
        # In visualization/sample_images/
        current_dir / "sample_images" / "IDRiD_046.jpg",
        current_dir / "sample_images" / "IDRiD_031.jpg",
        current_dir / "sample_images" / "IDRiD_001.jpg",
        current_dir / "sample_images" / "IDRiD_sample.jpg",

        # In main dataset
        project_root / "data/raw/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_046.jpg",
        project_root / "data/raw/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_031.jpg",
        project_root / "data/raw/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_001.jpg",

        # Alternative dataset structure
        project_root / "data/raw/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_046.jpg",
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

    # Try to find any JPG in training folders
    training_folders = [
        project_root / "data/raw/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set",
        project_root / "data/raw/B. Disease Grading/1. Original Images/a. Training Set",
    ]

    print("\n🔍 Searching for any JPG files in training folders...")
    for folder in training_folders:
        if folder.exists():
            jpg_files = list(folder.glob("*.jpg"))
            if jpg_files:
                selected_image = jpg_files[0]
                print(f"✅ Found JPG files in: {folder}")
                print(f"📸 Using: {selected_image.name}")
                return str(selected_image)

    return None


def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")

    dependencies = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'networkx': 'networkx',
        'sklearn': 'scikit-learn'
    }

    missing = []
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {module} ({package})")
        except ImportError:
            print(f"❌ {module} ({package}) - MISSING")
            missing.append(package)

    if missing:
        print(f"\n💡 Install missing packages:")
        print(f"pip install {' '.join(missing)}")
        return False

    return True


# Usage
if __name__ == "__main__":
    print("🕸️ Graph Structure Visualization Generator")
    print("=" * 50)

    # Check dependencies first
    if not check_dependencies():
        print("❌ Missing dependencies. Please install them first.")
        exit(1)

    # Try to find a sample image automatically
    image_path = find_sample_image()

    if image_path:
        print(f"📸 Using image: {Path(image_path).name}")

        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        visualize_graph_structure(image_path, "output/graph_structure_presentation.png")
    else:
        print("❌ No sample image found!")
        print("\n💡 Solutions:")
        print("1. Copy an IDRiD image to visualization/sample_images/")
        print("2. Or specify the correct path manually:")
        print("   image_path = 'path/to/your/image.jpg'")
        print("   visualize_graph_structure(image_path, 'output/graph_structure_presentation.png')")

        # Show what we're looking for
        current_dir = Path(__file__).parent
        sample_dir = current_dir / "sample_images"
        print(f"\n📁 Expected sample image locations:")
        print(f"   • {sample_dir / 'IDRiD_046.jpg'}")
        print(f"   • {sample_dir / 'IDRiD_031.jpg'}")
        print(f"   • {sample_dir / 'IDRiD_001.jpg'}")
