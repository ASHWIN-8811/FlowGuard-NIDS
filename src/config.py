"""
Configuration settings for Explainable Random Forest Network Flow IDS.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "CSE-CIC-IDS2018"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

# Ensure output directories exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Label mapping to high-level attack families (for 7-class grouped mode)
ATTACK_FAMILY_MAPPING = {
    "Benign": "Benign",
    "Bot": "Botnet",
    "Brute Force -Web": "Web Attack",
    "Brute Force -XSS": "Web Attack",
    "SQL Injection": "Web Attack",
    "SSH-Bruteforce": "Brute Force",
    "FTP-BruteForce": "Brute Force",
    "DDOS attack-HOIC": "DDoS",
    "DDOS attack-LOIC-UDP": "DDoS",
    "DDoS attacks-LOIC-HTTP": "DDoS",
    "DoS attacks-GoldenEye": "DoS",
    "DoS attacks-Hulk": "DoS",
    "DoS attacks-SlowHTTPTest": "DoS",
    "DoS attacks-Slowloris": "DoS",
    "Infilteration": "Infiltration",
}

# Target column name
TARGET_COLUMN = "Label"

# Default Model Hyperparameters
DEFAULT_RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
