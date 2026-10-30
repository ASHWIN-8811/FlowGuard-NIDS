"""
Data loading and stratified sampling utilities for CSE-CIC-IDS2018.
"""

import glob
import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from src.config import DATASET_DIR, TARGET_COLUMN, ATTACK_FAMILY_MAPPING, RANDOM_STATE

logger = logging.getLogger(__name__)


def list_parquet_files(dataset_dir: Path = DATASET_DIR) -> list:
    """Find all Parquet files in the dataset directory."""
    files = sorted(glob.glob(str(dataset_dir / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {dataset_dir}")
    return files


def load_stratified_dataset(
    dataset_dir: Path = DATASET_DIR,
    sample_size: Optional[int] = 120000,
    group_classes: bool = False,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Load dataset with smart stratified sampling.
    Retains 100% of minority attack classes (e.g., SQL Injection, FTP-BruteForce)
    and downsamples dominant classes (e.g., Benign, DDoS) to ensure balanced representation
    while maintaining realistic cloud traffic flow volume.
    """
    files = list_parquet_files(dataset_dir)
    logger.info("Found %d parquet files in %s", len(files), dataset_dir)

    dfs = []
    for f in files:
        logger.info("Reading %s...", Path(f).name)
        df_part = pd.read_parquet(f)
        dfs.append(df_part)

    df_full = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded full dataset: %d rows, %d columns", len(df_full), df_full.shape[1])

    # Strip column name whitespace if any
    df_full.columns = df_full.columns.str.strip()

    # Optional grouping into 7 attack families
    if group_classes:
        df_full[TARGET_COLUMN] = df_full[TARGET_COLUMN].map(
            lambda x: ATTACK_FAMILY_MAPPING.get(x, x)
        )
        logger.info("Grouped classes into attack families.")

    if sample_size is None or sample_size >= len(df_full):
        logger.info("Returning full dataset (%d rows).", len(df_full))
        return df_full

    # Smart stratified sampling
    class_counts = df_full[TARGET_COLUMN].value_counts()
    num_classes = len(class_counts)
    logger.info("Class distribution before sampling:\n%s", class_counts.to_string())

    sampled_dfs = []
    # For classes smaller than 5,000 samples, preserve all rows
    minority_threshold = 5000
    minority_classes = class_counts[class_counts <= minority_threshold].index.tolist()
    majority_classes = class_counts[class_counts > minority_threshold].index.tolist()

    # Retain all minority class records
    minority_count = 0
    for cls in minority_classes:
        sub_df = df_full[df_full[TARGET_COLUMN] == cls]
        sampled_dfs.append(sub_df)
        minority_count += len(sub_df)

    remaining_budget = max(sample_size - minority_count, 1000)
    # Distribute remaining budget across majority classes proportionally
    majority_total = class_counts[majority_classes].sum()

    rng = np.random.RandomState(random_state)
    for cls in majority_classes:
        sub_df = df_full[df_full[TARGET_COLUMN] == cls]
        prop = len(sub_df) / majority_total
        n_sample = int(np.round(remaining_budget * prop))
        # Ensure at least minority_threshold or min available is sampled
        n_sample = min(len(sub_df), max(minority_threshold, n_sample))
        sample_indices = rng.choice(sub_df.index, size=n_sample, replace=False)
        sampled_dfs.append(sub_df.loc[sample_indices])

    df_sampled = pd.concat(sampled_dfs, ignore_index=True).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)

    logger.info("Final sampled dataset: %d rows across %d classes", len(df_sampled), num_classes)
    logger.info("Class distribution after sampling:\n%s", df_sampled[TARGET_COLUMN].value_counts().to_string())
    return df_sampled


def get_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split into stratified train and test sets.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test
