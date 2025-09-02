#!/usr/bin/env python3
"""Training script for IDRiD hybrid model"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root / "src"))

from utils.config import load_config, create_directories, set_seed, validate_dataset_structure
from data_processing.dataset_loader import IDRiDDatasetLoader
from models.hybrid_model import IDRiDHybridModel

def setup_logging(log_dir: str, experiment_name: str):
    """Setup logging"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train IDRiD Hybrid Model')
    parser.add_argument('--config', default='configs/main_config.yaml')
    parser.add_argument('--experiment_name', default=None)
    parser.add_argument('--data_path', default=None)
    parser.add_argument('--output_dir', default='results/models')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"hybrid_training_{timestamp}"
    
    log_dir = os.path.join("results", "logs")
    logger = setup_logging(log_dir, args.experiment_name)
    
    logger.info("="*50)
    logger.info("IDRiD Hybrid Model Training")
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info("="*50)
    
    try:
        # Load configuration
        config = load_config(args.config)
        if args.data_path:
            config['dataset']['base_path'] = args.data_path
        
        set_seed(args.seed)
        create_directories(config)
        
        # Validate dataset
        dataset_path = config['dataset']['base_path']
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return
        
        if not validate_dataset_structure(dataset_path):
            logger.error("Invalid dataset structure")
            return
        
        # Load dataset
        dataset_loader = IDRiDDatasetLoader(dataset_path, task="grading")
        data = dataset_loader.load_data_for_task()
        
        train_images = data['train_images']
        train_labels = data['train_labels']
        
        logger.info(f"Training images: {len(train_images)}")
        logger.info(f"Training labels: {train_labels.shape}")
        
        # Prepare validation split
        val_images, val_labels = None, None
        if args.validate and len(train_images) > 10:
            val_split = int(0.2 * len(train_images))
            val_images = train_images[-val_split:]
            train_images = train_images[:-val_split]
            val_labels = train_labels.iloc[-val_split:].copy()
            train_labels = train_labels.iloc[:-val_split].copy()
        
        # Initialize and train model
        hybrid_model = IDRiDHybridModel(config)
        experiment_output_dir = os.path.join(args.output_dir, args.experiment_name)
        
        training_results = hybrid_model.train(
            train_image_paths=train_images,
            train_labels_df=train_labels,
            val_image_paths=val_images,
            val_labels_df=val_labels,
            save_dir=experiment_output_dir
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Results: {training_results}")
        
        # Save configuration
        import yaml
        config_save_path = os.path.join(experiment_output_dir, 'training_config.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Models saved to: {experiment_output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()