# Explainable Random Forest Framework for Multi-Class Intrusion Detection in Cloud Network Flow Logs

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework-Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit--Learn](https://img.shields.io/badge/Library-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![Explainability-SHAP](https://img.shields.io/badge/XAI-TreeSHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![Dataset-CSE--CIC--IDS2018](https://img.shields.io/badge/Dataset-CSE--CIC--IDS2018-informational.svg)](https://registry.opendata.aws/cse-cic-ids2018/)

An end-to-end Machine Learning and Explainable Artificial Intelligence (XAI) framework for multi-class cyber attack detection and instance-level attribution in tabular network flow data.

---

## ✍️ Authors

- **Ashwin V**
- **Mowlishwar T**

---

## 📌 About The Project

Modern networks generate vast amounts of bidirectional communication traffic every second. Because over 90% of contemporary network traffic is encrypted (HTTPS/TLS), traditional Deep Packet Inspection (DPI) methods fail to detect intrusions without decrypting private payloads.

This project implements a flow-based Network Intrusion Detection System (NIDS) that classifies network traffic solely using statistical connection telemetry (such as packet counts, inter-arrival times, byte rates, and TCP flags). 

To eliminate the "black-box" dilemma in cybersecurity, the framework couples an optimized multi-class **Random Forest** classifier with **TreeSHAP (SHapley Additive exPlanations)**. The system not only detects threats across 15 distinct classes with high accuracy but also mathematically explains **why** a specific flow was flagged, enabling swift and verifiable defensive actions.

### Key Capabilities:
- **15-Class Detection:** Distinguishes normal benign traffic from 14 distinct attack variants spanning DDoS, DoS, Botnets, Web Attacks, and Brute Force.
- **Explainable AI (XAI):** Generates local, instance-level SHAP waterfall charts for any flagged flow record and ranks overarching global decisive features.
- **Interactive Light-Mode Web Dashboard:** Built with Streamlit for single-click log upload, real-time threat KPIs, inspection tables, and downloadable reports.
- **Packaged Test Scenarios:** Comes pre-packaged with verified demonstration datasets (`demo_datasets/`) covering all attack categories for immediate testing.

---

## 📊 Recorded Model Evaluation Results

Evaluated on a held-out test split of **24,801 flow records** (`artifacts/test_samples.parquet`):

| Evaluation Metric | Recorded Value | Description |
| :--- | :---: | :--- |
| **Overall Accuracy** | **84.22%** | Proportion of total correctly classified network flows. |
| **Weighted F1-Score** | **0.8797** | F1-score weighted by support frequency across classes. |
| **Macro F1-Score** | **0.7367** | Unweighted harmonic mean of precision and recall. |
| **Macro Recall** | **0.8218** | Average detection sensitivity across all attack categories. |
| **Macro Precision** | **0.7067** | Average classification confidence across all classes. |
| **Active Features** | **69** | Statistical flow features retained after zero-variance pruning. |

---

## 🛠️ System Requirements

- **Operating System:** Windows 10/11, macOS, or Linux.
- **Python Version:** Python 3.10, 3.11, or 3.12.
- **RAM:** 8 GB minimum (16 GB recommended for running SHAP explanations).
- **Disk Space:** ~500 MB (pre-trained artifacts and demo datasets included).

### Dependencies (`requirements.txt`):
```text
pandas>=2.2.0
pyarrow>=15.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
scipy>=1.12.0
shap>=0.45.0
matplotlib>=3.8.0
seaborn>=0.13.0
streamlit>=1.35.0
joblib>=1.3.0
```

---

## 🚀 Quickstart: Installation & How to Run

### Step 1: Clone the Repository
Open your terminal (PowerShell, Command Prompt, or Bash) and clone the repository:
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### Step 2: Create & Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If script execution is restricted in PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

**On Windows (Command Prompt):**
```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Web Application
```bash
streamlit run app.py
```

Once executed, Streamlit will start a local server and automatically open the application in your web browser at:
```
http://localhost:8501
```

---

## 🖥️ How to Use the Project Dashboard

1. **Open the Dashboard:** Navigate to `http://localhost:8501` in your browser.
2. **Load a Test Dataset:**
   - Under **"📁 Packaged Demonstration Datasets"** on the right side of the main tab, select:  
     👉 `⚡ 1. All 15 Attack & Benign Scenarios (Combined Demo)`
   - Click the **"🚀 Load Selected Dataset"** button.
   - *(Alternatively, upload any custom network flow `.csv` or `.parquet` file using the drag-and-drop box).*
3. **Review Detection Analytics:**
   - View the top KPI summary cards (Total Flows, Benign Flows, Attacks Flagged, Threat Status).
   - Inspect the traffic proportion pie chart and the attack category horizontal bar chart.
4. **Filter & Inspect Records:**
   - Use the filter buttons to view *All Flows*, *Flagged Attacks Only*, or *Benign Only*.
   - Click **"📥 Download Annotated Predictions (CSV)"** to save the enriched security incident report.
5. **Generate Instance-Level SHAP Explainability:**
   - Scroll down to section **4. Instance-Level SHAP Explainability**.
   - Select any flow record index from the dropdown.
   - Click **"🔬 Compute SHAP Attribution for Flow"**.
   - The application computes exact TreeSHAP attributions and displays:
     - The top 5 features increasing the threat prediction score (red).
     - The top 5 features decreasing the threat prediction score (blue).
     - A horizontal SHAP waterfall bar chart.
     - An automated incident summary narrative explaining the technical reason for the classification.
6. **Explore Other Views:**
   - Use the left sidebar to switch to **Model Performance & Metrics** (confusion matrix, per-class breakdown), **Explainability & Feature Importance** (global SHAP vs. Gini rankings), or **Threat Taxonomy & Dataset Info**.

---

## 📁 Repository Structure

```
├── artifacts/                        # Pre-trained models, explainer metadata, and figures
│   ├── figures/                      # Normalized confusion matrix and performance charts
│   ├── explainer.joblib              # Serialized TreeSHAP background distribution
│   ├── feature_importances.csv       # Gini impurity feature ranking
│   ├── metrics.json                  # Complete recorded evaluation metrics
│   ├── preprocessor.joblib           # Fitted NetworkFlowPreprocessor object
│   ├── rf_model.joblib               # Trained Random Forest ensemble classifier
│   ├── shap_global.csv               # Global SHAP feature importance ranking
│   └── test_samples.parquet          # Held-out test split (24,801 flow records)
├── demo_datasets/                    # Ready-to-evaluate demonstration test logs
│   ├── 1_all_15_attack_scenarios.csv # Consolidated demo covering all 15 classes
│   ├── 2_normal_benign_traffic.csv   # Safe baseline benign enterprise flows
│   ├── 3_ddos_flood_incident.csv     # High-rate volumetric DDoS flood traffic
│   ├── 4_bruteforce_ssh_ftp.csv      # SSH and FTP credential brute-force flows
│   ├── 5_web_attacks_and_sqli.csv    # Web exploits (SQLi, XSS, Brute-Force Web)
│   ├── 6_mixed_enterprise_traffic.csv# Realistic multi-class enterprise traffic mix
│   ├── 7_raw_unlabeled_network_flows.csv # Raw unlabeled records for testing
│   └── 8_sample_network_flows.csv    # General representative flow sample
├── src/                              # Core Python source package
│   ├── __init__.py                   # Package initializer
│   ├── config.py                     # Centralized paths and hyperparameters
│   ├── data_loader.py                # Stratified data loading routines
│   ├── explainer.py                  # TreeSHAP explainer engine
│   ├── model.py                      # Random Forest model wrapper
│   ├── preprocessor.py               # Feature cleaning, imputation, and pruning
│   └── visualizer.py                 # Matplotlib / Seaborn visualization utilities
├── .streamlit/
│   └── config.toml                   # Streamlit light-theme configuration
├── app.py                            # Streamlit interactive light-mode web dashboard
├── evaluate.py                       # Standalone CLI evaluation script
├── train.py                          # End-to-end model training script
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules (excludes large datasets, venv, caches)
└── README.md                         # Project documentation
```

---

## 🧪 Additional CLI Commands

- **To run standalone evaluation on the test dataset:**
  ```bash
  python evaluate.py
  ```
- **To retrain the model from scratch:**
  ```bash
  python train.py
  ```
  *(Note: Retraining requires placing the official CSE-CIC-IDS2018 benchmark parquet files into a `CSE-CIC-IDS2018/` directory).*

---

## 📄 License & Attribution

This project is developed for educational and academic research purposes. The dataset utilized is the **CSE-CIC-IDS2018** benchmark provided collaboratively by the Communications Security Establishment (CSE) and the Canadian Institute for Cybersecurity (CIC), University of New Brunswick.
