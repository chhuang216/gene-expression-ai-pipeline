# File: src/train_rf.py
#!/usr/bin/env python
# coding: utf-8

# Imports & logging setup
import argparse                    # CLI parsing
import json                        # parse JSON params
import logging                     # logging messages
from typing import Dict, Optional  # type hints
import pandas as pd                # DataFrame handling
import joblib                      # model serialization
from sklearn.ensemble import RandomForestClassifier # Random Forest model
from sklearn.metrics import classification_report, roc_auc_score # evaluation metrics
from sklearn.model_selection import train_test_split # train/test split
from sklearn.preprocessing import StandardScaler # for scaling features

# Configure logger
type_check_fmt = "%(asctime)s %(levelname)s %(message)s" # format for type checking
logging.basicConfig(format=type_check_fmt, level=logging.INFO)
logger = logging.getLogger(__name__)


# Core training function
def train_rf(
    X: pd.DataFrame,                                            # Features DataFrame              
    y: pd.Series,                                     # Target Series       
    rf_params: Dict [str, Optional[float]]  # RandomForest parameters           
) -> RandomForestClassifier:                                                                        
    """
    Train a RandomForestClassifier with given parameters on provided data.

    Returns:
        Trained RandomForestClassifier.
    """
    logger.info("Initializing RandomForest with params: %s", rf_params)                         
    model = RandomForestClassifier(**rf_params)  # Initialize model with provided parameters                                                
    model.fit(X, y)     # Fit model to training data                                 
    logger.info("Training complete on %d samples.", X.shape[0])                                                             
    return model


# CLI / orchestration with train/test split
def main():
    parser = argparse.ArgumentParser(                                                   
        description="Train, evaluate, and save a RandomForestClassifier using a combined feature+label CSV." # Description for CLI
    )
    parser.add_argument(
        "--input-csv", required=True,
        help="CSV containing feature columns plus a 'label' column." # Input CSV path
    )
    parser.add_argument(
        "--rf-params", default="{}",
        help="JSON string of RF params, e.g., '{\"n_estimators\":100}'." # RF parameters in JSON format
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Proportion of data to reserve for test (0-1)." # Proportion of data for test set (default 20%)
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed for train-test split and model." # Random seed for reproducibility (default 42)
    )
    parser.add_argument(
        "--model-out", required=True,
        help="Path to save trained RF model (joblib .pkl)." # Output path for trained model
    )
    parser.add_argument(
        "--report-out", dest="report_out", default=None,
        help="Path to save classification report on test set (txt)." # Output path for classification report
    )
    parser.add_argument(
        "--scaler-out", default=None,
        help="Path to save fitted scaler (joblib .pkl)." # Output path for fitted scaler (optional, can be None)
    )
    args = parser.parse_args()

    # Load combined data
    df = pd.read_csv(args.input_csv, index_col=0)     # Load CSV into DataFrame                                                       
    if 'label' not in df.columns:                                   
        logger.error("Input CSV must contain a 'label' column.")
        raise SystemExit(1)                                 

    # Split features and target
    y = df['label'] # Target variable
    X = df.drop(columns='label')    # Features DataFrame        

    # Parse RF params
    try:
        rf_params = json.loads(args.rf_params)      # Parse JSON string into dictionary     
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON for --rf-params: %s", e)                                 
        raise

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,           # Features and target       
        test_size=args.test_size,   # Proportion for test set               
        random_state=args.random_state, # Random seed for reproducibility           
        stratify=y  # Stratify by target to maintain class distribution     
    )
    logger.info("Data split: %d train, %d test samples.", X_train.shape[0], X_test.shape[0])                            

    # Fit and save scaler
    if args.scaler_out:
        scaler = StandardScaler() # Initialize scaler
        scaler.fit(X_train) # Fit scaler on training data
        joblib.dump(scaler, args.scaler_out) # Save fitted scaler to file
        logger.info("Saved scaler to %s", args.scaler_out)  # Save scaler to specified path
    
    # Train model
    model = train_rf(X_train, y_train, rf_params)      # Train the RandomForestClassifier with training data and parameters                              

    # Evaluate on test set
    y_pred = model.predict(X_test)  # Predict on test set
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None # Predict probabilities if available
    report = classification_report(y_test, y_pred)     # Generate classification report on test set                         
    logger.info("Test Classification Report:\n%s", report)    # Log classification report                          
    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)     # Calculate ROC AUC score if probabilities are available                       
        logger.info("Test ROC AUC Score: %.4f", auc)                        
    else:
        auc = None          

    # Save report if requested
    if args.report_out:
        with open(args.report_out, 'w') as f:     # Open report file for writing  
            f.write("Random Forest - Classification Report on Test Set:\n" + report) # Write classification report to file
            if auc is not None:
                f.write(f"\nROC AUC: {auc:.4f}\n") # Write ROC AUC score to file
        logger.info("Saved test report to %s", args.report_out)

    # Save the trained model
    joblib.dump(model, args.model_out) # Save trained model to specified path
    logger.info("Saved model to %s", args.model_out)


if __name__ == "__main__":
    main()
