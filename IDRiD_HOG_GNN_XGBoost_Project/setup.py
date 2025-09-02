from setuptools import setup, find_packages

setup(
    name="idrid-hog-gnn-xgboost",
    version="1.0.0",
    description="Hybrid model for diabetic retinopathy detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "torch-geometric>=2.3.0",
        "xgboost>=1.7.0",
        "scikit-learn>=1.3.0",
        "opencv-python>=4.8.0",
        "streamlit>=1.25.0"
    ]
)