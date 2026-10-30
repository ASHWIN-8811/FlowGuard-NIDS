"""
Random Forest Classifier training, evaluation, and serialization.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src.config import DEFAULT_RF_PARAMS, ARTIFACTS_DIR

logger = logging.getLogger(__name__)


class CloudFlowRandomForest:
    """
    Random Forest framework wrapper for network flow intrusion detection.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = {**DEFAULT_RF_PARAMS, **(params or {})}
        self.model = RandomForestClassifier(**self.params)
        self.is_trained = False

    def train(self, X_train: pd.DataFrame, y_train: np.ndarray) -> "CloudFlowRandomForest":
        logger.info(
            "Training RandomForestClassifier with %d estimators, max_depth=%s...",
            self.params.get("n_estimators", 100),
            self.params.get("max_depth", None),
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Random Forest model training completed.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        return self.model.predict_proba(X)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        target_names: list,
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation across multi-class metrics.
        """
        logger.info("Evaluating model on %d test samples...", len(X_test))
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        p_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        r_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # ROC-AUC (one-vs-rest) if all classes are present in y_test
        roc_auc = None
        try:
            roc_auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
        except Exception as e:
            logger.warning("Could not compute ROC-AUC: %s", e)

        cm = confusion_matrix(y_test, y_pred).tolist()
        report_dict = classification_report(
            y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0
        )
        report_str = classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
        )

        metrics = {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "roc_auc_macro_ovr": roc_auc,
            "confusion_matrix": cm,
            "classification_report": report_dict,
            "classification_report_str": report_str,
            "classes": target_names,
        }

        logger.info("Accuracy: %.4f | Macro F1: %.4f | Weighted F1: %.4f", acc, f1_macro, f1_weighted)
        return metrics

    def get_feature_importances(self, feature_names: list) -> pd.DataFrame:
        """
        Extract Gini feature importances sorted in descending order.
        """
        importances = self.model.feature_importances_
        df_fi = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        ).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        return df_fi

    def save(self, filepath: Optional[Path] = None) -> Path:
        target_path = filepath or (ARTIFACTS_DIR / "rf_model.joblib")
        joblib.dump(self.model, target_path)
        logger.info("Model saved successfully to %s", target_path)
        return target_path

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "CloudFlowRandomForest":
        target_path = filepath or (ARTIFACTS_DIR / "rf_model.joblib")
        instance = cls()
        instance.model = joblib.load(target_path)
        instance.is_trained = True
        logger.info("Model loaded successfully from %s", target_path)
        return instance
