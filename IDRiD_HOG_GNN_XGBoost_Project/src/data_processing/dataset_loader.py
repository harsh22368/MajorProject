"""IDRiD Dataset Loader with nested folder support"""
import os
import pandas as pd
import cv2
import numpy as np
import glob
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class IDRiDDatasetLoader:
    """IDRiD dataset loader with support for nested folder structure"""
    
    def __init__(self, base_path: str, task: str = "grading"):
        self.base_path = base_path
        self.task = task
        
        if not self._validate_structure():
            raise ValueError(f"Invalid dataset structure at {base_path}")
        
        logger.info(f"Dataset loader initialized for {task}")
    
    def _validate_structure(self) -> bool:
        """Validate dataset structure with nested folder support"""
        required = ["A. Segmentation", "B. Disease Grading", "C. Localization"]
        return all(os.path.exists(os.path.join(self.base_path, p)) for p in required)
    
    def _find_nested_path(self, base_folder: str, sub_path: str) -> str:
        """Find path considering nested folder structure"""
        # Try nested structure first (e.g., B. Disease Grading/B. Disease Grading/)
        nested_path = os.path.join(self.base_path, base_folder, base_folder, sub_path)
        if os.path.exists(nested_path):
            return nested_path
        
        # Try direct structure (e.g., B. Disease Grading/)
        direct_path = os.path.join(self.base_path, base_folder, sub_path)
        if os.path.exists(direct_path):
            return direct_path
        
        return None
    
    def load_grading_data(self) -> Tuple[List[str], pd.DataFrame, List[str], pd.DataFrame]:
        """Load disease grading data with nested folder support"""
        
        # Find training images
        train_path = self._find_nested_path("B. Disease Grading", "1. Original Images/a. Training Set")
        if not train_path:
            raise FileNotFoundError("Training images not found")
        
        train_images = sorted(glob.glob(os.path.join(train_path, "*.jpg")))
        
        # Find test images
        test_path = self._find_nested_path("B. Disease Grading", "1. Original Images/b. Testing Set")
        test_images = sorted(glob.glob(os.path.join(test_path, "*.jpg"))) if test_path else []
        
        # Find labels directory
        labels_path = self._find_nested_path("B. Disease Grading", "2. Groundtruths")
        if not labels_path:
            raise FileNotFoundError("Groundtruth labels not found")
        
        # Load training labels
        train_labels_file = None
        test_labels_file = None
        
        for file in os.listdir(labels_path):
            if 'training' in file.lower() and file.endswith('.csv'):
                train_labels_file = os.path.join(labels_path, file)
            elif 'testing' in file.lower() and file.endswith('.csv'):
                test_labels_file = os.path.join(labels_path, file)
        
        train_labels = pd.read_csv(train_labels_file) if train_labels_file else pd.DataFrame()
        test_labels = pd.read_csv(test_labels_file) if test_labels_file else pd.DataFrame()
        
        logger.info(f"Loaded {len(train_images)} train, {len(test_images)} test images")
        logger.info(f"Labels found at: {labels_path}")
        return train_images, train_labels, test_images, test_labels
    
    def load_image(self, image_path: str, target_size: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
        """Load and preprocess image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        return image.astype(np.float32) / 255.0
    
    def get_image_id(self, image_path: str) -> str:
        """Extract image ID"""
        return os.path.splitext(os.path.basename(image_path))[0]
    
    def load_data_for_task(self) -> Dict:
        """Load data for specified task"""
        if self.task == "grading":
            train_imgs, train_labels, test_imgs, test_labels = self.load_grading_data()
            return {
                'train_images': train_imgs,
                'train_labels': train_labels,
                'test_images': test_imgs,
                'test_labels': test_labels
            }
        else:
            raise ValueError(f"Task {self.task} not implemented")