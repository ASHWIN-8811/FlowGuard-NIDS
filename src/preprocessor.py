"""
Preprocessing pipeline for Network Flow Data.
Handles inf/nan values, zero-variance columns, and label encodings.
"""

import logging
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class NetworkFlowPreprocessor:
    """
    Robust Preprocessor tailored for CICFlowMeter network flow data:
    - Cleans non-numeric and whitespace column headers
    - Replaces infinite values (+/- inf)
    - Imputes missing values with column medians
    - Removes zero-variance (constant) features
    - Encodes multi-class string labels
    """

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.dropped_features: List[str] = []
        self.feature_names: List[str] = []
        self.medians: dict = {}
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "NetworkFlowPreprocessor":
        logger.info("Fitting NetworkFlowPreprocessor on %d rows, %d columns", X.shape[0], X.shape[1])
        X_clean = X.copy()

        # Clean column names
        X_clean.columns = [c.strip() for c in X_clean.columns]

        # Replace +/- inf with NaN
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)

        # Identify zero-variance features (std == 0 or all NaN or single unique value)
        for col in X_clean.columns:
            if X_clean[col].nunique(dropna=True) <= 1:
                self.dropped_features.append(col)

        cols_to_keep = [c for c in X_clean.columns if c not in self.dropped_features]
        logger.info("Identified %d zero-variance columns to drop: %s", len(self.dropped_features), self.dropped_features)

        # Compute medians for imputation
        for col in cols_to_keep:
            median_val = float(X_clean[col].median(skipna=True))
            if np.isnan(median_val):
                median_val = 0.0
            self.medians[col] = median_val

        self.feature_names = cols_to_keep

        # Fit label encoder if y is provided
        if y is not None:
            self.label_encoder.fit(y)
            logger.info("Fitted LabelEncoder with %d classes: %s", len(self.label_encoder.classes_), list(self.label_encoder.classes_))

        self.is_fitted = True
        return self

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet.")

        X_clean = X.copy()
        X_clean.columns = [c.strip() for c in X_clean.columns]

        # Drop columns identified as zero-variance during fit
        drop_existing = [c for c in self.dropped_features if c in X_clean.columns]
        X_clean = X_clean.drop(columns=drop_existing, errors="ignore")

        # Ensure all expected features are present
        for col in self.feature_names:
            if col not in X_clean.columns:
                X_clean[col] = self.medians.get(col, 0.0)

        # Reorder to match fit feature order
        X_clean = X_clean[self.feature_names]

        # Replace inf with nan
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)

        # Impute missing values with fitted medians
        X_clean = X_clean.fillna(self.medians)

        # Ensure numeric float32 for speed and memory efficiency
        X_clean = X_clean.astype(np.float32)
        return X_clean

    def transform_labels(self, y: pd.Series) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet.")
        return self.label_encoder.transform(y)

    def inverse_transform_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        return self.label_encoder.inverse_transform(y_encoded)

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        self.fit(X, y)
        X_transformed = self.transform_features(X)
        y_transformed = self.transform_labels(y)
        return X_transformed, y_transformed
