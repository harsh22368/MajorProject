"""XGBoost classifier for DR and DME prediction"""
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
import os

logger = logging.getLogger(__name__)

class XGBoostClassifier:
    """XGBoost classifier for DR/DME prediction"""
    
    def __init__(self, config: Dict, task: str = "dr"):
        self.config = config
        self.xgb_config = config['xgboost']
        self.task = task
        
        if task == "dr":
            self.num_classes = len(config['dataset']['tasks']['grading']['dr_grades'])
        elif task == "dme":
            self.num_classes = len(config['dataset']['tasks']['grading']['dme_risk'])
        else:
            raise ValueError(f"Unknown task: {task}")
        
        self.model = xgb.XGBClassifier(
            n_estimators=self.xgb_config['n_estimators'],
            max_depth=self.xgb_config['max_depth'],
            learning_rate=self.xgb_config['learning_rate'],
            subsample=self.xgb_config['subsample'],
            colsample_bytree=self.xgb_config['colsample_bytree'],
            random_state=self.xgb_config['random_state'],
            objective='multi:softmax' if self.num_classes > 2 else 'binary:logistic',
            num_class=self.num_classes if self.num_classes > 2 else None,
            use_label_encoder=False
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        logger.info(f"XGBoost initialized for {task} with {self.num_classes} classes")
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train XGBoost classifier"""
        logger.info(f"Training XGBoost for {self.task}")
        
        X_scaled = self.scaler.fit_transform(X)
        
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_val_scaled, y_val)]
        
        self.model.fit(X_scaled, y, eval_set=eval_set, verbose=False)
        self.is_fitted = True
        
        # Calculate training accuracy
        train_pred = self.predict(X)
        train_accuracy = accuracy_score(y, train_pred)
        
        results = {
            'train_accuracy': train_accuracy,
            'best_iteration': getattr(self.model, 'best_iteration', self.model.n_estimators)
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            results['val_accuracy'] = val_accuracy
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save_model(self, save_path: str) -> None:
        """Save trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'task': self.task,
            'num_classes': self.num_classes,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, save_path)
    
    def load_model(self, load_path: str) -> None:
        """Load trained model"""
        model_data = joblib.load(load_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_fitted = model_data.get('is_fitted', True)

class MultiTaskXGBoost:
    """Multi-task XGBoost for DR and DME"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dr_classifier = XGBoostClassifier(config, task="dr")
        self.dme_classifier = XGBoostClassifier(config, task="dme")
    
    def fit(self, X: np.ndarray, y_dr: np.ndarray, y_dme: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val_dr: Optional[np.ndarray] = None,
            y_val_dme: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train both classifiers"""
        dr_results = self.dr_classifier.fit(X, y_dr, X_val, y_val_dr)
        dme_results = self.dme_classifier.fit(X, y_dme, X_val, y_val_dme)
        
        return {
            'dr_results': dr_results,
            'dme_results': dme_results
        }
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict both tasks"""
        return {
            'dr_predictions': self.dr_classifier.predict(X),
            'dme_predictions': self.dme_classifier.predict(X),
            'dr_probabilities': self.dr_classifier.predict_proba(X),
            'dme_probabilities': self.dme_classifier.predict_proba(X)
        }
    
    def save_models(self, save_dir: str) -> None:
        """Save both models"""
        self.dr_classifier.save_model(os.path.join(save_dir, 'xgb_dr_model.pkl'))
        self.dme_classifier.save_model(os.path.join(save_dir, 'xgb_dme_model.pkl'))
    
    def load_models(self, save_dir: str) -> None:
        """Load both models"""
        self.dr_classifier.load_model(os.path.join(save_dir, 'xgb_dr_model.pkl'))
        self.dme_classifier.load_model(os.path.join(save_dir, 'xgb_dme_model.pkl'))