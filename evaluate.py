"""
Evaluation script to test the trained model on saved test data or custom parquet files.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from src.config import ARTIFACTS_DIR, FIGURES_DIR
from src.visualizer import plot_confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def evaluate_saved_model(test_path: Path):
    model_path = ARTIFACTS_DIR / "rf_model.joblib"
    prep_path = ARTIFACTS_DIR / "preprocessor.joblib"

    if not model_path.exists() or not prep_path.exists():
        logger.error("Trained model or preprocessor not found in %s. Please run train.py first.", ARTIFACTS_DIR)
        sys.exit(1)

    logger.info("Loading model and preprocessor...")
    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)

    logger.info("Loading evaluation dataset from %s...", test_path)
    if str(test_path).endswith(".parquet"):
        df_test = pd.read_parquet(test_path)
    else:
        df_test = pd.read_csv(test_path)

    if "True_Label" in df_test.columns:
        y_raw = df_test["True_Label"]
        X_raw = df_test.drop(columns=["True_Label"])
    elif "Label" in df_test.columns:
        y_raw = df_test["Label"]
        X_raw = df_test.drop(columns=["Label"])
    else:
        logger.error("Could not find 'Label' or 'True_Label' column in test data.")
        sys.exit(1)

    X_proc = preprocessor.transform_features(X_raw)
    y_true = preprocessor.transform_labels(y_raw)

    y_pred = model.predict(X_proc)
    class_names = list(preprocessor.label_encoder.classes_)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 55)
    print(f"EVALUATION RESULTS ON: {test_path.name}")
    print("=" * 55)
    print(f"Accuracy:         {acc:.4f}")
    print(f"Macro F1-Score:   {f1_macro:.4f}")
    print(f"Weighted F1-Score:{f1_weighted:.4f}")
    print("\nDetailed Classification Breakdown:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("=" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Network Flow IDS model")
    parser.add_argument(
        "--test_file",
        type=str,
        default=str(ARTIFACTS_DIR / "test_samples.parquet"),
        help="Path to parquet or csv test data file",
    )
    args = parser.parse_args()
    evaluate_saved_model(Path(args.test_file))
