"""Configuration utilities for IDRiD project"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "configs/main_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories"""
    paths = config.get('paths', {})
    directories = [
        paths.get('models', 'results/models'),
        paths.get('figures', 'results/figures'),
        paths.get('logs', 'results/logs'),
        'data/processed/hog_features',
        'data/processed/graph_data',
        'data/processed/gnn_embeddings'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def get_device():
    """Get best available device"""
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def validate_dataset_structure(base_path: str) -> bool:
    """Validate IDRiD dataset structure with nested folder support"""
    required_paths = [
        # Handle nested structure: A. Segmentation/A. Segmentation/
        ["A. Segmentation/A. Segmentation/1. Original Images/a. Training Set",
         "A. Segmentation/1. Original Images/a. Training Set"],
        
        # Handle nested structure: B. Disease Grading/B. Disease Grading/
        ["B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set",
         "B. Disease Grading/1. Original Images/a. Training Set"],
        
        ["B. Disease Grading/B. Disease Grading/1. Original Images/b. Testing Set",
         "B. Disease Grading/1. Original Images/b. Testing Set"],
         
        ["B. Disease Grading/B. Disease Grading/2. Groundtruths",
         "B. Disease Grading/2. Groundtruths"],
         
        # Handle nested structure: C. Localization/C. Localization/
        ["C. Localization/C. Localization/1. Original Images/a. Training Set",
         "C. Localization/1. Original Images/a. Training Set"]
    ]
    
    for path_options in required_paths:
        path_found = False
        for path in path_options:
            full_path = os.path.join(base_path, path)
            if os.path.exists(full_path):
                path_found = True
                break
        
        if not path_found:
            print(f"Missing required paths: {path_options}")
            return False
    
    return True