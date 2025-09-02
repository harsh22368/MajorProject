#!/usr/bin/env python3
"""Evaluation script for trained models"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root / "src"))

from utils.config import load_config
from data_processing.dataset_loader import IDRiDDatasetLoader
from models.hybrid_model import IDRiDHybridModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Evaluate IDRiD Hybrid Model')
    parser.add_argument('--config', default='configs/main_config.yaml')
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--data_path', default=None)
    parser.add_argument('--output_dir', default='results/evaluation')
    
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
        if args.data_path:
            config['dataset']['base_path'] = args.data_path
        
        # Load dataset
        dataset_loader = IDRiDDatasetLoader(config['dataset']['base_path'], task="grading")
        data = dataset_loader.load_data_for_task()
        
        test_images = data['test_images']
        test_labels = data['test_labels']
        
        if len(test_images) == 0:
            logger.error("No test images found")
            return
        
        # Load trained model
        hybrid_model = IDRiDHybridModel(config)
        hybrid_model.load_models(args.model_dir)
        
        # Evaluate model (simplified version)
        logger.info(f"Evaluating on {len(test_images)} test images")
        
        # Load test images
        test_image_list = []
        for img_path in test_images[:10]:  # Evaluate on first 10 for demo
            try:
                import cv2
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    test_image_list.append(img)
            except Exception as e:
                logger.warning(f"Error loading {img_path}: {e}")
        
        # Extract features and make predictions
        if test_image_list:
            features = hybrid_model.extract_features_batch(test_image_list)
            predictions = hybrid_model.xgb_classifier.predict(features)
            
            logger.info("Evaluation completed!")
            logger.info(f"Sample predictions: {predictions['dr_predictions'][:5]}")
            logger.info(f"Feature shape: {features.shape}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()