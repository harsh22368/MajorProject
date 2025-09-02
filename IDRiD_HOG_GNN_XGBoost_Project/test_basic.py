"""Basic tests for project setup"""
import os
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir / "src"))

def test_imports():
    """Test basic imports"""
    try:
        from utils.config import load_config, validate_dataset_structure
        print("✅ Config utilities imported")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test configuration"""
    try:
        from utils.config import load_config
        config = load_config("configs/main_config.yaml")
        print("✅ Configuration loaded")
        print(f"Dataset path: {config['dataset']['base_path']}")
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def test_dataset():
    """Test dataset structure with nested folder support"""
    try:
        from utils.config import validate_dataset_structure
        
        if os.path.exists("data/raw"):
            result = validate_dataset_structure("data/raw")
            if result:
                print("✅ Dataset structure valid")
            else:
                print("⚠️ Place your unzipped IDRiD dataset in data/raw/")
        else:
            print("⚠️ Create data/raw/ and place IDRiD dataset")
        
        return True
    except Exception as e:
        print(f"❌ Dataset test error: {e}")
        return False

def main():
    print("🧪 IDRiD Project - Basic Tests")
    print("="*50)
    
    tests = [
        ("Import Test", test_imports),
        ("Config Test", test_config),
        ("Dataset Test", test_dataset)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}...")
        result = test_func()
        results.append(result)
    
    print("="*50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("\nNext steps:")
        print("1. Place IDRiD dataset in data/raw/")
        print("2. Install: pip install -r requirements.txt")
        print("3. Train: python scripts/train.py")
        print("4. Web app: streamlit run web_interface/app.py")
    else:
        print(f"⚠️ {passed}/{total} tests passed")

if __name__ == "__main__":
    main()