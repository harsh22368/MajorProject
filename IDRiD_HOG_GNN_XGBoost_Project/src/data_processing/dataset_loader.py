"""IDRiD Dataset Loader with nested folder support and robust CSV handling"""
import os
import pandas as pd
import cv2
import numpy as np
import glob
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class IDRiDDatasetLoader:
    """IDRiD dataset loader with support for nested folder structure and safe CSV loading"""

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
        """Load disease grading data with robust CSV handling"""

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

        # Find CSV files
        train_labels_file = None
        test_labels_file = None

        for file in os.listdir(labels_path):
            if 'training' in file.lower() and file.endswith('.csv'):
                train_labels_file = os.path.join(labels_path, file)
            elif 'testing' in file.lower() and file.endswith('.csv'):
                test_labels_file = os.path.join(labels_path, file)

        # Load training labels with robust encoding handling
        if train_labels_file:
            try:
                train_labels = pd.read_csv(train_labels_file, encoding='utf-8')
                logger.info(f"Loaded training labels with UTF-8 encoding")
            except UnicodeDecodeError:
                logger.warning("UTF-8 failed, trying latin-1 encoding")
                train_labels = pd.read_csv(train_labels_file, encoding='latin-1')

            # Clean the labels
            train_labels = self._clean_labels_dataframe(train_labels)
        else:
            train_labels = pd.DataFrame()

        # Load test labels with robust encoding handling
        if test_labels_file:
            try:
                test_labels = pd.read_csv(test_labels_file, encoding='utf-8')
                logger.info(f"Loaded test labels with UTF-8 encoding")
            except UnicodeDecodeError:
                logger.warning("UTF-8 failed, trying latin-1 encoding")
                test_labels = pd.read_csv(test_labels_file, encoding='latin-1')

            # Clean the labels
            test_labels = self._clean_labels_dataframe(test_labels)
        else:
            test_labels = pd.DataFrame()

        logger.info(f"Loaded {len(train_images)} train, {len(test_images)} test images")
        logger.info(f"Labels found at: {labels_path}")
        return train_images, train_labels, test_images, test_labels

    def _clean_labels_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean labels DataFrame to remove corrupted data"""
        if df.empty:
            return df

        logger.info(f"Cleaning labels DataFrame with {len(df)} entries")

        # Clean DR grade column if it exists
        if 'Retinopathy grade' in df.columns:
            logger.info("Cleaning 'Retinopathy grade' column")

            # Convert to numeric, replacing invalid entries with NaN
            df['Retinopathy grade'] = pd.to_numeric(
                df['Retinopathy grade'], errors='coerce'
            )

            # Count invalid entries before cleaning
            invalid_count = df['Retinopathy grade'].isna().sum()
            if invalid_count > 0:
                logger.warning(f"Found {invalid_count} invalid DR grade entries")

            # Remove rows with NaN grades
            df = df.dropna(subset=['Retinopathy grade'])

            # Convert to integer
            df['Retinopathy grade'] = df['Retinopathy grade'].astype(int)

            # Keep only valid DR grades (0-4)
            valid_grades = (df['Retinopathy grade'] >= 0) & (df['Retinopathy grade'] <= 4)
            invalid_grade_count = len(df) - valid_grades.sum()

            if invalid_grade_count > 0:
                logger.warning(f"Removing {invalid_grade_count} entries with invalid grade ranges")

            df = df[valid_grades]

            logger.info(f"Cleaned labels: {len(df)} valid entries remaining")

        # Clean DME risk column if it exists
        if 'Risk of macular edema ' in df.columns:  # Note the trailing space in IDRiD dataset
            logger.info("Cleaning 'Risk of macular edema' column")

            df['Risk of macular edema '] = pd.to_numeric(
                df['Risk of macular edema '], errors='coerce'
            )

            # Fill NaN with 0 for DME risk
            df['Risk of macular edema '] = df['Risk of macular edema '].fillna(0).astype(int)

            # Keep only valid DME risk (0-2)
            df = df[
                (df['Risk of macular edema '] >= 0) &
                (df['Risk of macular edema '] <= 2)
                ]

        # Clean Image name column - remove any non-printable characters
        if 'Image name' in df.columns:
            df['Image name'] = df['Image name'].astype(str).str.strip()
            # Remove any entries with corrupted image names
            df = df[df['Image name'].str.len() > 0]

        # Reset index after cleaning
        df = df.reset_index(drop=True)

        logger.info(f"Final cleaned DataFrame has {len(df)} entries")
        return df

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
