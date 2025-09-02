"""HOG Feature Extractor for IDRiD images"""
import cv2
import numpy as np
from skimage.feature import hog
import joblib
import os
from typing import Tuple, List, Dict, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class IDRiDHOGExtractor:
    """HOG feature extractor for retinal images"""
    
    def __init__(self, config: Dict):
        self.hog_config = config['hog']
        self.dataset_config = config['dataset']
        
        self.orientations = self.hog_config['orientations']
        self.pixels_per_cell = tuple(self.hog_config['pixels_per_cell'])
        self.cells_per_block = tuple(self.hog_config['cells_per_block'])
        self.block_norm = self.hog_config['block_norm']
        self.patch_size = self.hog_config['patch_size']
        self.stride = self.hog_config['stride']
        self.target_size = tuple(self.dataset_config['processed_size'])
        
        logger.info("HOG Extractor initialized")
    
    def extract_patch_hog_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract HOG features from image patches"""
        # Convert to grayscale
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image.copy()
        
        # Resize
        img_resized = cv2.resize(image_gray, self.target_size)
        img_resized = img_resized.astype(np.float32) / 255.0
        
        patches = []
        coordinates = []
        height, width = img_resized.shape
        
        for y in range(0, height - self.patch_size + 1, self.stride):
            for x in range(0, width - self.patch_size + 1, self.stride):
                patch = img_resized[y:y+self.patch_size, x:x+self.patch_size]
                
                if np.var(patch) < 0.01:
                    continue
                
                try:
                    hog_desc = hog(
                        patch,
                        orientations=self.orientations,
                        pixels_per_cell=self.pixels_per_cell,
                        cells_per_block=self.cells_per_block,
                        block_norm=self.block_norm,
                        feature_vector=True,
                        transform_sqrt=True
                    )
                    
                    patches.append(hog_desc)
                    coordinates.append([x / width, y / height])
                except Exception as e:
                    logger.warning(f"HOG extraction failed at ({x}, {y}): {e}")
                    continue
        
        if len(patches) == 0:
            dummy_hog = np.zeros((1, self._calculate_hog_dim()))
            dummy_coords = np.array([[0.5, 0.5]])
            return dummy_hog, dummy_coords
        
        return np.array(patches), np.array(coordinates)
    
    def _calculate_hog_dim(self) -> int:
        """Calculate HOG feature dimension"""
        try:
            dummy_patch = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            dummy_hog = hog(
                dummy_patch,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                feature_vector=True
            )
            return len(dummy_hog)
        except:
            # Fallback calculation
            cells_x = self.patch_size // self.pixels_per_cell[0]
            cells_y = self.patch_size // self.pixels_per_cell[1]
            blocks_x = cells_x - self.cells_per_block[0] + 1
            blocks_y = cells_y - self.cells_per_block[1] + 1
            return blocks_x * blocks_y * self.cells_per_block[0] * self.cells_per_block[1] * self.orientations