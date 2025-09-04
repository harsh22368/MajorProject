"""XGBoost classifier for DR and DME prediction with robust input validation"""
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
    """XGBoost classifier with input validation for DR/DME prediction"""

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

    def _validate_and_clean_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and clean input data to remove corrupted entries"""
        logger.info(f"Validating and cleaning data: X shape {X.shape}, y shape {y.shape}")

        # Convert to numpy arrays
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        # Clean X - replace NaN, inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        # Clean y - ensure it's numeric and valid
        try:
            # Convert to numeric, replacing non-numeric with NaN
            y_numeric = pd.to_numeric(pd.Series(y), errors='coerce')
            y = y_numeric.values
        except Exception as e:
            logger.warning(f"Error converting labels to numeric: {e}")
            y = y.astype(str)
            # Remove non-numeric characters
            y_clean = []
            for label in y:
                # Extract digits only
                clean_label = ''.join(filter(str.isdigit, str(label)))
                if clean_label:
                    y_clean.append(int(clean_label))
                else:
                    y_clean.append(0)  # Default to class 0
            y = np.array(y_clean)

        # Ensure y is integer type
        y = y.astype(int)

        # Validate label ranges
        if self.task == "dr":
            valid_mask = (y >= 0) & (y <= 4)  # DR grades 0-4
        elif self.task == "dme":
            valid_mask = (y >= 0) & (y <= 2)  # DME risk 0-2
        else:
            valid_mask = (y >= 0) & (y < self.num_classes)

        # Count invalid entries
        invalid_count = np.sum(~valid_mask)
        if invalid_count > 0:
            logger.warning(f"Removing {invalid_count} samples with invalid labels")

        # Filter valid samples
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        # Remove samples with all-zero features (likely corrupted)
        feature_sum = np.sum(np.abs(X_clean), axis=1)
        non_zero_mask = feature_sum > 1e-6

        zero_feature_count = np.sum(~non_zero_mask)
        if zero_feature_count > 0:
            logger.warning(f"Removing {zero_feature_count} samples with zero features")

        X_final = X_clean[non_zero_mask]
        y_final = y_clean[non_zero_mask]

        logger.info(f"Data cleaned: {len(X_final)} valid samples remaining")

        if len(X_final) == 0:
            raise ValueError("No valid samples remaining after cleaning!")

        return X_final, y_final

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train XGBoost classifier with robust input validation"""
        logger.info(f"Training XGBoost for {self.task}")

        # Clean training data
        X_clean, y_clean = self._validate_and_clean_data(X, y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)

        # Handle validation data if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            try:
                X_val_clean, y_val_clean = self._validate_and_clean_data(X_val, y_val)
                X_val_scaled = self.scaler.transform(X_val_clean)
                eval_set = [(X_val_scaled, y_val_clean)]
            except Exception as e:
                logger.warning(f"Validation data cleaning failed: {e}. Training without validation.")
                eval_set = None

        # Train model with error handling
        try:
            self.model.fit(X_scaled, y_clean, eval_set=eval_set, verbose=False)
            self.is_fitted = True
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            # Try with simpler parameters
            logger.info("Retrying with simpler XGBoost parameters")

            simple_model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                objective='multi:softmax' if self.num_classes > 2 else 'binary:logistic',
                num_class=self.num_classes if self.num_classes > 2 else None,
                use_label_encoder=False
            )

            simple_model.fit(X_scaled, y_clean, verbose=False)
            self.model = simple_model
            self.is_fitted = True

        # Calculate training accuracy
        train_pred = self.predict(X_clean)
        train_accuracy = accuracy_score(y_clean, train_pred)

        results = {
            'train_accuracy': train_accuracy,
            'best_iteration': getattr(self.model, 'best_iteration', self.model.n_estimators),
            'num_samples': len(X_clean)
        }

        # Calculate validation accuracy if available
        if eval_set is not None:
            try:
                val_pred = self.predict(X_val_clean)
                val_accuracy = accuracy_score(y_val_clean, val_pred)
                results['val_accuracy'] = val_accuracy
                results['num_val_samples'] = len(X_val_clean)
            except:
                logger.warning("Could not calculate validation accuracy")

        logger.info(f"Training completed. Accuracy: {train_accuracy:.4f}")
        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with input validation"""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")

        # Clean input data
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        # Scale features
        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with input validation"""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")

        # Clean input data
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        # Scale features
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
        logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path: str) -> None:
        """Load trained model"""
        model_data = joblib.load(load_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_fitted = model_data.get('is_fitted', True)
        logger.info(f"Model loaded from {load_path}")


class MultiTaskXGBoost:
    """Multi-task XGBoost for DR and DME with robust error handling"""

    def __init__(self, config: Dict):
        self.config = config
        self.dr_classifier = XGBoostClassifier(config, task="dr")
        self.dme_classifier = XGBoostClassifier(config, task="dme")

    def fit(self, X: np.ndarray, y_dr: np.ndarray, y_dme: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val_dr: Optional[np.ndarray] = None,
            y_val_dme: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train both classifiers with error handling"""
        logger.info("Training multi-task XGBoost (DR + DME)")

        results = {}

        # Train DR classifier
        try:
            dr_results = self.dr_classifier.fit(X, y_dr, X_val, y_val_dr)
            results['dr_results'] = dr_results
            logger.info(f"DR classifier trained successfully. Accuracy: {dr_results['train_accuracy']:.4f}")
        except Exception as e:
            logger.error(f"DR classifier training failed: {e}")
            results['dr_results'] = {'train_accuracy': 0.0, 'error': str(e)}

        # Train DME classifier
        try:
            dme_results = self.dme_classifier.fit(X, y_dme, X_val, y_val_dme)
            results['dme_results'] = dme_results
            logger.info(f"DME classifier trained successfully. Accuracy: {dme_results['train_accuracy']:.4f}")
        except Exception as e:
            logger.error(f"DME classifier training failed: {e}")
            results['dme_results'] = {'train_accuracy': 0.0, 'error': str(e)}

        return results

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict both tasks with error handling"""
        results = {}

        try:
            results['dr_predictions'] = self.dr_classifier.predict(X)
            results['dr_probabilities'] = self.dr_classifier.predict_proba(X)
        except Exception as e:
            logger.error(f"DR prediction failed: {e}")
            results['dr_predictions'] = np.zeros(len(X), dtype=int)
            results['dr_probabilities'] = np.ones((len(X), 5)) / 5  # Uniform distribution

        try:
            results['dme_predictions'] = self.dme_classifier.predict(X)
            results['dme_probabilities'] = self.dme_classifier.predict_proba(X)
        except Exception as e:
            logger.error(f"DME prediction failed: {e}")
            results['dme_predictions'] = np.zeros(len(X), dtype=int)
            results['dme_probabilities'] = np.ones((len(X), 3)) / 3  # Uniform distribution

        return results

    def save_models(self, save_dir: str) -> None:
        """Save both models"""
        os.makedirs(save_dir, exist_ok=True)

        try:
            self.dr_classifier.save_model(os.path.join(save_dir, 'xgb_dr_model.pkl'))
        except Exception as e:
            logger.error(f"Failed to save DR model: {e}")

        try:
            self.dme_classifier.save_model(os.path.join(save_dir, 'xgb_dme_model.pkl'))
        except Exception as e:
            logger.error(f"Failed to save DME model: {e}")

    def load_models(self, save_dir: str) -> None:
        """Load both models"""
        try:
            self.dr_classifier.load_model(os.path.join(save_dir, 'xgb_dr_model.pkl'))
        except Exception as e:
            logger.error(f"Failed to load DR model: {e}")

        try:
            self.dme_classifier.load_model(os.path.join(save_dir, 'xgb_dme_model.pkl'))
        except Exception as e:
            logger.error(f"Failed to load DME model: {e}")
