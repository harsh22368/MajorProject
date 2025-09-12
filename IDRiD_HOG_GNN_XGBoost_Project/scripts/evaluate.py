#!/usr/bin/env python3
"""Evaluation with Confusion Matrix and ROC/PR plots for DR & DME."""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to PYTHONPATH
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root / "src"))

from utils.config import load_config
from data_processing.dataset_loader import IDRiDDatasetLoader
from models.hybrid_model import IDRiDHybridModel

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DR_CLASSES = ["No-DR", "Mild", "Moderate", "Severe", "PDR"]
DME_CLASSES = ["No-DME", "Mild DME", "Severe DME"]

# ---------- Plot helpers ----------
def save_confusion_matrix(y_true, y_pred, class_names, title, out_path):
    # Normalize by true labels to show per-class accuracy
    cm = confusion_matrix(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        normalize="true"
    )
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f")  # sklearn API
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def save_roc_ovr(y_true, y_proba, class_names, title, out_path):
    # One-vs-Rest ROC curves with macro AUC (official approach)
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    aucs = []
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title} (macro AUC={np.mean(aucs):.3f})")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def save_pr_ovr(y_true, y_proba, class_names, title, out_path):
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    aps = []
    for i, name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_bin[:, i], y_proba[:, i])
        aps.append(ap)
        ax.plot(recall, precision, lw=1.5, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title} (macro AP={np.mean(aps):.3f})")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# ---------- Column resolver ----------
def pick_col_df(df, candidates):
    """
    Robustly pick a column from a pandas DataFrame, tolerating spaces/case.
    Returns numpy array or None.
    """
    # First strip leading/trailing spaces from headers (fixes: 'Risk of macular edema ')
    df = df.rename(columns=lambda c: c.strip())
    # Build lowercase lookup
    lowmap = {c.lower(): c for c in df.columns}
    for name in candidates:
        key = name.strip().lower()
        if key in lowmap:
            return df[lowmap[key]].to_numpy()
    return None

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Evaluate IDRiD Hybrid Model")
    parser.add_argument("--config", default="configs/main_config.yaml")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--output_dir", default="results/evaluation")
    parser.add_argument("--limit", type=int, default=0, help="limit test images (0=all)")
    args = parser.parse_args()

    try:
        # Load config and dataset
        config = load_config(args.config)
        if args.data_path:
            config["dataset"]["base_path"] = args.data_path

        dataset_loader = IDRiDDatasetLoader(config["dataset"]["base_path"], task="grading")
        data = dataset_loader.load_data_for_task()

        test_images = data.get("test_images", [])
        test_labels = data.get("test_labels")

        if test_labels is None:
            raise KeyError("test_labels not returned by dataset loader")

        # Resolve DR/DME arrays from DataFrame or dict
        if hasattr(test_labels, "columns"):
            # pandas DataFrame case (print once for debug)
            print("test_labels columns:", list(test_labels.columns))
            test_dr = pick_col_df(test_labels, ["Retinopathy grade", "retinopathy_grade", "dr"])
            test_dme = pick_col_df(test_labels, ["Risk of macular edema", "Risk of macular edema ", "risk_of_macular_edema", "dme"])
        elif isinstance(test_labels, dict):
            test_dr = test_labels.get("Retinopathy grade") or test_labels.get("retinopathy_grade") or test_labels.get("dr")
            test_dme = test_labels.get("Risk of macular edema") or test_labels.get("risk_of_macular_edema") or test_labels.get("dme")
        elif isinstance(test_labels, (list, tuple)) and len(test_labels) == 2:
            test_dr, test_dme = test_labels
        else:
            raise KeyError("Unsupported structure for test_labels")

        if test_dr is None or test_dme is None:
            raise KeyError(f"Could not map DR/DME labels. Available: {list(test_labels.columns) if hasattr(test_labels,'columns') else list(test_labels.keys())}")

        if len(test_images) == 0:
            logger.error("No test images found")
            return

        # Optional limit
        if args.limit and args.limit > 0:
            test_images = test_images[:args.limit]
            test_dr = test_dr[:args.limit]
            test_dme = test_dme[:args.limit]

        # Load model
        hybrid_model = IDRiDHybridModel(config)
        hybrid_model.load_models(args.model_dir)

        # Read images
        import cv2
        imgs = []
        for p in test_images:
            img = cv2.imread(p)
            if img is None:
                logger.warning(f"Could not read: {p}")
                continue
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not imgs:
            logger.error("No readable test images.")
            return

        # Extract embeddings and predict probabilities
        feats = hybrid_model.extract_features_batch(imgs)

        # Your classifier must provide probability outputs
        # If you only have predict(), add these methods in src/models/xgboost_classifier.py
        dr_proba = hybrid_model.xgb_classifier.predict_proba_dr(feats)   # [N,5]
        dme_proba = hybrid_model.xgb_classifier.predict_proba_dme(feats) # [N,3]
        dr_pred = np.argmax(dr_proba, axis=1)
        dme_pred = np.argmax(dme_proba, axis=1)

        # Save figures
        out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        save_confusion_matrix(test_dr, dr_pred, DR_CLASSES, "DR Confusion Matrix (Normalized)", out_dir / "cm_dr.png")
        save_confusion_matrix(test_dme, dme_pred, DME_CLASSES, "DME Confusion Matrix (Normalized)", out_dir / "cm_dme.png")
        save_roc_ovr(test_dr, dr_proba, DR_CLASSES, "DR ROC (OvR)", out_dir / "roc_dr.png")
        save_roc_ovr(test_dme, dme_proba, DME_CLASSES, "DME ROC (OvR)", out_dir / "roc_dme.png")
        save_pr_ovr(test_dr, dr_proba, DR_CLASSES, "DR Precision–Recall (OvR)", out_dir / "pr_dr.png")
        save_pr_ovr(test_dme, dme_proba, DME_CLASSES, "DME Precision–Recall (OvR)", out_dir / "pr_dme.png")

        logger.info(f"Saved Figure M5 assets to: {out_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
