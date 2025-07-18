# src/pipeline.py
#!/usr/bin/env python
# coding: utf-8
""" 
# Gene Expression AI Pipeline
# This script orchestrates the entire pipeline from preprocessing to model training and explanation.
"""

import subprocess # for running shell commands
import logging # for logging
import os # for directory management

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def run_step(description, command):
    logging.info(f"== Running: {description} ===")
    try:
        subprocess.run(command, check=True) # run the command and check for errors
    except subprocess.CalledProcessError:
        logging.error(f"Step failed: {description}") 
        raise RuntimeError(f"Failed: {description}") 

def main():
    
    # Make sure all of our output dirs are there
    for d in [
        "src/data",
        "src/figures",
        "src/models",
        "src/scalers",
        "src/reports"
    ]:
        os.makedirs(d, exist_ok=True)
    
    # Define the steps in the pipeline
    steps = [
        ("Preprocess",
        ["python", "src/preprocess.py",
        "--series-matrix", "data/GSE2034_series_matrix.txt.gz", 
        "--family-soft", "data/GSE2034_family.soft.gz",
        "--out-csv", "src/data/filtered_expression_data.csv",
        "--plot-out", "src/figures/elbow_plot.png"]),

        ("Feature Selection",
        ["python", "src/feature_selection.py",
        "--input-csv", "src/data/filtered_expression_data.csv",
        "--k", "20",
        "--out-csv", "src/data/selected_expression_data.csv",
        "--plot-out", "src/figures/top20_importances.png"]),

        ("Train Random Forest",
        ["python", "src/train_rf.py",
        "--input-csv", "src/data/selected_expression_data.csv",
        "--test-size", "0.2",
        "--random-state", "42",
        "--model-out", "src/models/rf_model.pkl",
        "--scaler-out", "src/scalers/scaler_rf.pkl",
        "--report-out", "src/reports/rf_report.txt"]),

        ("Train XGBoost",
        ["python", "src/train_xgb.py",
        "--input-csv", "src/data/selected_expression_data.csv",
        "--xgb-params", '{"n_estimators":100,"max_depth":5}',
        "--test-size", "0.2",
        "--random-state", "42",
        "--model-out", "src/models/xgb_model.pkl",
        "--scaler-out", "src/scalers/scaler_xgb.pkl",
        "--report-out", "src/reports/xgb_report.txt"]),

        ("Train DNN",
        ["python", "src/train_dnn.py",
        "--input-csv", "src/data/selected_expression_data.csv",
        "--model-out", "src/models/dnn_model.keras",
        "--report-out", "src/reports/dnn_report.txt",
        "--loss-plot-out", "src/figures/loss_plot.png",
        "--matrix-plot-out", "src/figures/confusion_matrix.png",
        "--scaler-out", "src/scalers/scaler_dnn.pkl",
        "--test-size", "0.2",
        "--random-state", "42",
        "--max-iter", "25",
        "--epochs", "200",
        "--patient", "5"]),

        ("Explain SHAP", [
        "python", "src/explain_shap.py",
        "--input-csv", "src/data/selected_expression_data.csv", 
        "--models-dir", "src/models",
        "--scalers-dir", "src/scalers",
        "--figures-dir", "src/figures",
        "--test-size", "0.2",
        "--random-state", "42",
        "--show-plot"]),
    ]

    for desc, cmd in steps:
        run_step(desc, cmd)

if __name__ == "__main__":
    main()
