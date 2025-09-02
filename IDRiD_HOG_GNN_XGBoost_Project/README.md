# IDRiD HOG+GNN+XGBoost Project

Complete hybrid approach for diabetic retinopathy detection using HOG features, Graph Neural Networks, and XGBoost on the IDRiD dataset.

## 🚀 Quick Start

1. **Unzip this project**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Place your unzipped IDRiD dataset in `data/raw/`:**
   ```
   data/raw/
   ├── A. Segmentation/
   ├── B. Disease Grading/
   └── C. Localization/
   ```
4. **Test setup:**
   ```bash
   python test_basic.py
   ```
5. **Train model:**
   ```bash
   python scripts/train.py
   ```
6. **Launch web interface:**
   ```bash
   streamlit run web_interface/app.py
   ```

## 📁 Architecture

- **HOG**: Interpretable feature extraction from retinal patches
- **GNN**: Spatial relationship modeling via graph networks  
- **XGBoost**: Robust final classification with feature importance

## 🎯 Features

- Multi-task DR grading (0-4) and DME risk assessment (0-2)
- Clinical web interface with treatment recommendations
- Handles nested dataset folder structures automatically
- CPU-optimized processing (no GPU required)
- Complete logging and experiment tracking

Built specifically for the IDRiD dataset structure with nested folder support.
