"""
Visualization utilities for model metrics, confusion matrix, and SHAP explanations.
"""

import logging
from pathlib import Path
from typing import List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plot generation
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    normalize: bool = True,
) -> Path:
    """
    Plot and save confusion matrix heatmap.
    Blanks out 0 values so the diagonal and true misclassifications stand out clearly.
    """
    plt.figure(figsize=(14, 11))
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype("float") / row_sums

    # Build custom annotation matrix: leave zeros blank!
    annot_array = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            pct = cm_norm[i, j]
            if count == 0:
                annot_array[i, j] = ""  # Blank for 0!
            else:
                if normalize:
                    annot_array[i, j] = f"{pct * 100:.1f}%\n({count:,})"
                else:
                    annot_array[i, j] = f"{count:,}"

    display_data = cm_norm if normalize else cm
    sns.heatmap(
        display_data,
        annot=annot_array,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 8.5, "weight": "semibold"},
        linewidths=0.5,
        linecolor="#E2E8F0",
    )
    title = (
        "Normalized Confusion Matrix with Sample Counts (CSE-CIC-IDS2018)"
        if normalize
        else "Confusion Matrix Flow Counts (CSE-CIC-IDS2018)"
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("True Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    plt.tight_layout()

    out_file = save_path or (FIGURES_DIR / "confusion_matrix.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info("Saved confusion matrix plot to %s", out_file)
    return out_file


def plot_feature_importances(
    df_importances: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[Path] = None,
    title: str = "Top Random Forest Feature Importances (MDI)",
) -> Path:
    """
    Plot and save top feature importances bar chart.
    """
    plt.figure(figsize=(10, 8))
    top_df = df_importances.head(top_n).iloc[::-1]  # Reverse for top-down display

    colors = sns.color_palette("viridis", len(top_df))
    plt.barh(top_df["Feature"], top_df.iloc[:, 1], color=colors)
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Network Flow Feature", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_file = save_path or (FIGURES_DIR / "feature_importances.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info("Saved feature importances plot to %s", out_file)
    return out_file


def plot_class_performance(
    classification_report_dict: dict,
    save_path: Optional[Path] = None,
) -> Path:
    """
    Plot Precision, Recall, and F1-Score breakdown for each intrusion class.
    """
    records = []
    for cls, metrics in classification_report_dict.items():
        if cls in ["accuracy", "macro avg", "weighted avg"]:
            continue
        records.append({
            "Class": cls,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-Score": metrics["f1-score"],
            "Support": metrics["support"],
        })

    df_metrics = pd.DataFrame(records)
    if df_metrics.empty:
        return Path("")

    plt.figure(figsize=(14, 7))
    df_melt = pd.melt(
        df_metrics,
        id_vars=["Class"],
        value_vars=["Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score",
    )

    ax = sns.barplot(data=df_melt, x="Class", y="Score", hue="Metric", palette="Set2")
    plt.title("Per-Class Detection Performance (Precision / Recall / F1)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Network Flow Class", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Metric", loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_file = save_path or (FIGURES_DIR / "per_class_performance.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info("Saved per-class performance plot to %s", out_file)
    return out_file
