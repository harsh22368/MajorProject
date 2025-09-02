"""Complete Hybrid Model"""
import torch
import numpy as np
import pandas as pd
import cv2
import os
import logging
from typing import Dict, List, Any, Optional, Tuple

from src.data_processing.hog_extractor import IDRiDHOGExtractor
from src.data_processing.graph_builder import IDRiDGraphBuilder
from src.models.gnn_model import GNNWithFeatureExtraction
from src.models.xgboost_classifier import MultiTaskXGBoost
from src.utils.config import get_device

logger = logging.getLogger(__name__)

class IDRiDHybridModel:
    """Complete hybrid model for DR detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = get_device()
        
        # Initialize components
        self.hog_extractor = IDRiDHOGExtractor(config)
        self.graph_builder = IDRiDGraphBuilder(config)
        self.gnn_model = GNNWithFeatureExtraction(config).to(self.device)
        self.xgb_classifier = MultiTaskXGBoost(config)
        
        self.is_trained = False
        self.feature_dim = None
        
        logger.info("Hybrid Model initialized")
    
    def extract_features_from_image(self, image: np.ndarray) -> np.ndarray:
        """Extract features from single image"""
        try:
            # Extract HOG features
            hog_features, coordinates = self.hog_extractor.extract_patch_hog_features(image)
            
            if len(hog_features) == 0:
                return np.zeros(64)
            
            # Build graph
            graph = self.graph_builder.build_spatial_graph(hog_features, coordinates)
            graph = graph.to(self.device)
            
            # Extract GNN features
            self.gnn_model.eval()
            with torch.no_grad():
                gnn_features = self.gnn_model(graph)
                gnn_features = gnn_features.cpu().numpy().flatten()
            
            return gnn_features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return np.zeros(64)
    
    def extract_features_batch(self, image_list: List[np.ndarray]) -> np.ndarray:
        """Extract features from batch of images"""
        features_list = []
        
        for image in image_list:
            features = self.extract_features_from_image(image)
            features_list.append(features)
        
        return np.array(features_list)
    
    def prepare_training_data(self, image_paths: List[str], 
                            labels_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data"""
        images = []
        valid_image_ids = []
        
        for img_path in image_paths:
            try:
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    image_id = os.path.splitext(os.path.basename(img_path))[0]
                    valid_image_ids.append(image_id)
            except Exception as e:
                logger.warning(f"Error loading {img_path}: {e}")
                continue
        
        # Extract features
        features = self.extract_features_batch(images)
        
        # Organize labels
        dr_labels = []
        dme_labels = []
        
        for image_id in valid_image_ids:
            label_row = labels_df[labels_df.iloc[:, 0] == image_id]
            
            if len(label_row) > 0:
                dr_grade = int(label_row.iloc[0, 1])
                dme_risk = int(label_row.iloc[0, 2])
                dr_labels.append(dr_grade)
                dme_labels.append(dme_risk)
            else:
                dr_labels.append(0)
                dme_labels.append(0)
        
        return features, np.array(dr_labels), np.array(dme_labels)
    
    def train(self, train_image_paths: List[str], train_labels_df: pd.DataFrame,
              val_image_paths: Optional[List[str]] = None,
              val_labels_df: Optional[pd.DataFrame] = None,
              save_dir: Optional[str] = None) -> Dict[str, Any]:
        """Train complete hybrid model"""
        logger.info("Starting hybrid model training")
        
        # Prepare training data
        train_features, train_dr_labels, train_dme_labels = self.prepare_training_data(
            train_image_paths, train_labels_df
        )
        
        # Prepare validation data if provided
        val_features, val_dr_labels, val_dme_labels = None, None, None
        if val_image_paths and val_labels_df is not None:
            val_features, val_dr_labels, val_dme_labels = self.prepare_training_data(
                val_image_paths, val_labels_df
            )
        
        self.feature_dim = train_features.shape[1]
        
        # Train XGBoost classifiers
        xgb_results = self.xgb_classifier.fit(
            train_features, train_dr_labels, train_dme_labels,
            val_features, val_dr_labels, val_dme_labels
        )
        
        self.is_trained = True
        
        if save_dir:
            self.save_models(save_dir)
        
        return {
            'xgb_results': xgb_results,
            'train_samples': len(train_image_paths),
            'feature_dim': self.feature_dim
        }
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Make predictions on single image"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        features = self.extract_features_from_image(image)
        features = features.reshape(1, -1)
        
        predictions = self.xgb_classifier.predict(features)
        
        dr_confidence = np.max(predictions['dr_probabilities'])
        dme_confidence = np.max(predictions['dme_probabilities'])
        
        dr_grade = predictions['dr_predictions'][0]
        clinical_rec = self._get_clinical_recommendation(dr_grade, dr_confidence)
        
        return {
            'dr_grade': int(dr_grade),
            'dr_confidence': float(dr_confidence),
            'dme_risk': int(predictions['dme_predictions'][0]),
            'dme_confidence': float(dme_confidence),
            'dr_probabilities': predictions['dr_probabilities'][0].tolist(),
            'dme_probabilities': predictions['dme_probabilities'][0].tolist(),
            'clinical_recommendation': clinical_rec
        }
    
    def _get_clinical_recommendation(self, dr_grade: int, confidence: float) -> Dict[str, str]:
        """Get clinical recommendation"""
        recommendations = {
            0: {"action": "Routine annual screening", "timeline": "12 months", "urgency": "Low"},
            1: {"action": "Increased monitoring", "timeline": "6 months", "urgency": "Medium"}, 
            2: {"action": "Ophthalmologist referral", "timeline": "3-4 months", "urgency": "Medium-High"},
            3: {"action": "Urgent referral required", "timeline": "2 weeks", "urgency": "High"},
            4: {"action": "Immediate treatment required", "timeline": "1 week", "urgency": "Critical"}
        }
        
        rec = recommendations.get(dr_grade, recommendations[0])
        
        if confidence < 0.7:
            rec["note"] = "Low confidence - manual review recommended"
        
        return rec
    
    def save_models(self, save_dir: str) -> None:
        """Save all models"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save GNN model
        gnn_path = os.path.join(save_dir, 'gnn_model.pth')
        torch.save(self.gnn_model.state_dict(), gnn_path)
        
        # Save XGBoost models
        self.xgb_classifier.save_models(save_dir)
        
        # Save metadata
        import joblib
        metadata = {
            'config': self.config,
            'is_trained': self.is_trained,
            'feature_dim': self.feature_dim,
            'device': str(self.device)
        }
        joblib.dump(metadata, os.path.join(save_dir, 'model_metadata.pkl'))
    
    def load_models(self, save_dir: str) -> None:
        """Load all models"""
        import joblib
        
        # Load metadata
        metadata = joblib.load(os.path.join(save_dir, 'model_metadata.pkl'))
        self.config = metadata['config']
        self.is_trained = metadata['is_trained']
        self.feature_dim = metadata['feature_dim']
        
        # Load GNN model
        gnn_path = os.path.join(save_dir, 'gnn_model.pth')
        self.gnn_model.load_state_dict(torch.load(gnn_path, map_location=self.device))
        
        # Load XGBoost models
        self.xgb_classifier.load_models(save_dir)