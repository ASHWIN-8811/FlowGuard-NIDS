"""
End-to-End Training and Explainability Pipeline for Network Flow IDS.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.config import ARTIFACTS_DIR, FIGURES_DIR, RANDOM_STATE
from src.data_loader import load_stratified_dataset, get_train_test_data
from src.preprocessor import NetworkFlowPreprocessor
from src.model import CloudFlowRandomForest
from src.explainer import CloudIDSExplainer
from src.visualizer import (
    plot_confusion_matrix,
    plot_feature_importances,
    plot_class_performance,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_pipeline(
    sample_size: int = 100000,
    n_estimators: int = 100,
    max_depth: int = 20,
    group_classes: bool = False,
    shap_samples: int = 300,
):
    logger.info("=" * 70)
    logger.info("Starting Explainable Random Forest Network Flow IDS Training Pipeline")
    logger.info("=" * 70)

    # 1. Load Data
    effective_sample = sample_size if sample_size > 0 else None
    df = load_stratified_dataset(
        sample_size=effective_sample,
        group_classes=group_classes,
        random_state=RANDOM_STATE,
    )

    # 2. Train-Test Split
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = get_train_test_data(
        df, test_size=0.2, random_state=RANDOM_STATE
    )

    # 3. Preprocessing
    preprocessor = NetworkFlowPreprocessor()
    X_train, y_train = preprocessor.fit_transform(X_train_raw, y_train_raw)
    X_test = preprocessor.transform_features(X_test_raw)
    y_test = preprocessor.transform_labels(y_test_raw)

    class_names = list(preprocessor.label_encoder.classes_)
    logger.info("Classes (%d): %s", len(class_names), class_names)

    # 4. Model Training
    rf_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth if max_depth > 0 else None,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    rf_wrapper = CloudFlowRandomForest(params=rf_params)
    rf_wrapper.train(X_train, y_train)

    # 5. Evaluation
    metrics = rf_wrapper.evaluate(X_test, y_test, target_names=class_names)
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT:")
    print("=" * 50)
    print(metrics["classification_report_str"])
    print("=" * 50 + "\n")

    # 6. Feature Importances (Gini)
    df_importances = rf_wrapper.get_feature_importances(preprocessor.feature_names)
    df_importances.to_csv(ARTIFACTS_DIR / "feature_importances.csv", index=False)
    plot_feature_importances(
        df_importances,
        top_n=20,
        save_path=FIGURES_DIR / "feature_importances.png",
        title="Top 20 Random Forest Feature Importances (Gini/MDI)",
    )

    # 7. Confusion Matrix and Performance Plots
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, class_names, save_path=FIGURES_DIR / "confusion_matrix.png")
    plot_class_performance(
        metrics["classification_report"],
        save_path=FIGURES_DIR / "per_class_performance.png",
    )

    # 8. SHAP Explainability Engine
    logger.info("Initializing SHAP Explainability Engine...")
    explainer = CloudIDSExplainer(
        model=rf_wrapper.model,
        feature_names=preprocessor.feature_names,
        class_names=class_names,
    )
    explainer.fit_explainer(X_train, sample_size=min(150, len(X_train)))

    # Compute SHAP on a test subset for global attribution
    n_shap = min(shap_samples, len(X_test))
    shap_subset = X_test.sample(n=n_shap, random_state=RANDOM_STATE)
    shap_values = explainer.compute_shap_values(shap_subset)
    df_shap_global = explainer.get_global_importance(shap_values, shap_subset)
    df_shap_global.to_csv(ARTIFACTS_DIR / "shap_global.csv", index=False)

    plot_feature_importances(
        df_shap_global,
        top_n=20,
        save_path=FIGURES_DIR / "shap_importance.png",
        title="Top 20 Decisive Features via SHAP (Mean |SHAP Value|)",
    )

    # 9. Save Artifacts
    rf_wrapper.save(ARTIFACTS_DIR / "rf_model.joblib")
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")
    explainer.save(ARTIFACTS_DIR / "explainer.joblib")

    # Save test sample dataset with true labels for interactive evaluation in Streamlit
    sample_eval_df = X_test_raw.copy()
    sample_eval_df["True_Label"] = y_test_raw.values
    sample_eval_df.head(5000).to_parquet(ARTIFACTS_DIR / "test_samples.parquet", index=False)

    # Save metrics JSON (omitting non-serializable objects)
    metrics_export = {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
        "roc_auc_macro_ovr": metrics["roc_auc_macro_ovr"],
        "classes": metrics["classes"],
        "classification_report": metrics["classification_report"],
        "sample_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "num_features": len(preprocessor.feature_names),
        "dropped_features": preprocessor.dropped_features,
    }
    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_export, f, indent=2)

    logger.info("Pipeline completed successfully! All artifacts stored in %s", ARTIFACTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Explainable Random Forest for Network Flow IDS"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100000,
        help="Total sample size with minority-preserving stratification (0 for full dataset)",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of decision trees in Random Forest",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=20,
        help="Maximum depth of trees",
    )
    parser.add_argument(
        "--group_classes",
        action="store_true",
        help="Group 15 attack classes into 7 major threat families",
    )
    parser.add_argument(
        "--shap_samples",
        type=int,
        default=250,
        help="Number of test samples for global SHAP calculation",
    )

    args = parser.parse_args()
    run_pipeline(
        sample_size=args.sample_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        group_classes=args.group_classes,
        shap_samples=args.shap_samples,
    )
