# 🚀 Setup Instructions

## 1. Unzip This Project
Extract the ZIP file to your desired location (e.g., D:\Python Projects\)

## 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## 3. Add Your Dataset
Place your unzipped IDRiD dataset folders directly in `data/raw/`:
```
data/raw/
├── A. Segmentation/
├── B. Disease Grading/
└── C. Localization/
```

**Note:** The code handles nested folder structures automatically.

## 4. Test Setup
```bash
python test_basic.py
```

## 5. Train Model
```bash
python scripts/train.py
```

## 6. Launch Web Interface
```bash
streamlit run web_interface/app.py
```

Visit http://localhost:8501 for the clinical interface.

## 🎯 Key Features
- Handles your exact dataset structure with nested folders
- No manual path configuration needed
- Complete error handling and validation
- Professional clinical interface ready
