"""
Explainability framework using SHAP (SHapley Additive exPlanations)
for Multi-Class Network Flow Intrusion Detection.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import shap
import joblib

from src.config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)


class CloudIDSExplainer:
    """
    SHAP-based Explainability framework for Random Forest Network Flow Models:
    - Fast TreeExplainer computation
    - Global feature importance attribution
    - Local (per-flow) instance-level explanations
    - Human-readable alert synthesis
    """

    def __init__(self, model, feature_names: List[str], class_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.explainer = None
        self.shap_values = None
        self.sample_background = None

    def fit_explainer(self, background_data: pd.DataFrame, sample_size: int = 200) -> "CloudIDSExplainer":
        """
        Fit SHAP TreeExplainer using a representative sample of background cloud flows.
        """
        logger.info("Initializing SHAP TreeExplainer with %d background samples...", min(sample_size, len(background_data)))
        if len(background_data) > sample_size:
            self.sample_background = background_data.sample(n=sample_size, random_state=42)
        else:
            self.sample_background = background_data.copy()

        # TreeExplainer is heavily optimized for scikit-learn RandomForest
        self.explainer = shap.TreeExplainer(self.model, data=self.sample_background)
        logger.info("SHAP TreeExplainer initialized successfully.")
        return self

    def compute_shap_values(self, X: pd.DataFrame) -> Any:
        """
        Compute SHAP values for a given evaluation set.
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit_explainer() first.")

        logger.info("Computing SHAP values for %d samples...", len(X))
        shap_vals = self.explainer.shap_values(X)
        self.shap_values = shap_vals
        return shap_vals

    def get_global_importance(self, shap_values: Any, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute global mean absolute SHAP value across all classes for each feature.
        """
        # SHAP returns either a list of [n_samples, n_features] for each class
        # or a 3D numpy array [n_samples, n_features, n_classes]
        if isinstance(shap_values, list):
            # List of arrays per class
            mean_abs_per_class = [np.abs(sv).mean(axis=0) for sv in shap_values]
            overall_mean_abs = np.mean(mean_abs_per_class, axis=0)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # 3D array: (samples, features, classes)
            overall_mean_abs = np.abs(shap_values).mean(axis=(0, 2))
        else:
            overall_mean_abs = np.abs(shap_values).mean(axis=0)

        df_global = pd.DataFrame({
            "Feature": self.feature_names,
            "Mean_Abs_SHAP": overall_mean_abs
        }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

        return df_global

    def explain_instance(
        self,
        instance_row: pd.Series,
        predicted_class_idx: int,
        confidence: float,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Provide local SHAP attribution and human-readable explanation for a single cloud flow.
        """
        if self.explainer is None:
            raise ValueError("Explainer is not initialized.")

        instance_df = pd.DataFrame([instance_row], columns=self.feature_names)
        shap_out = self.explainer.shap_values(instance_df)

        # Extract SHAP array for the target predicted class
        if isinstance(shap_out, list):
            class_shap = shap_out[predicted_class_idx][0]
        elif isinstance(shap_out, np.ndarray) and shap_out.ndim == 3:
            class_shap = shap_out[0, :, predicted_class_idx]
        else:
            class_shap = shap_out[0]

        # Combine features with their values and SHAP contributions
        contributions = []
        for feat, val, sv in zip(self.feature_names, instance_row.values, class_shap):
            contributions.append({
                "feature": feat,
                "value": float(val),
                "shap_value": float(sv),
                "abs_shap": abs(float(sv)),
            })

        df_contrib = pd.DataFrame(contributions).sort_values(by="abs_shap", ascending=False)
        top_positive = df_contrib[df_contrib["shap_value"] > 0].head(top_k).to_dict("records")
        top_negative = df_contrib[df_contrib["shap_value"] < 0].head(top_k).to_dict("records")

        pred_label = self.class_names[predicted_class_idx]

        # Construct natural language explanation for SOC analysts
        reasoning = [
            f"Network flow classified as '{pred_label}' with {confidence * 100:.1f}% confidence."
        ]
        if top_positive:
            top_factors = [
                f"{item['feature']}={item['value']:.2f} (+{item['shap_value']:.3f})"
                for item in top_positive[:3]
            ]
            reasoning.append(f"Key indicators increasing threat likelihood: {', '.join(top_factors)}.")
        if top_negative:
            suppressors = [
                f"{item['feature']}={item['value']:.2f} ({item['shap_value']:.3f})"
                for item in top_negative[:2]
            ]
            reasoning.append(f"Counter-balancing benign patterns: {', '.join(suppressors)}.")

        return {
            "predicted_class": pred_label,
            "predicted_class_index": predicted_class_idx,
            "confidence": float(confidence),
            "top_positive_features": top_positive,
            "top_negative_features": top_negative,
            "human_readable_summary": " ".join(reasoning),
            "all_contributions": df_contrib.to_dict("records"),
        }

    def save(self, filepath: Optional[Path] = None) -> Path:
        target_path = filepath or (ARTIFACTS_DIR / "explainer.joblib")
        # Save necessary state for quick reconstruction in dashboard
        state = {
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "sample_background": self.sample_background,
        }
        joblib.dump(state, target_path)
        logger.info("Explainer metadata saved to %s", target_path)
        return target_path
