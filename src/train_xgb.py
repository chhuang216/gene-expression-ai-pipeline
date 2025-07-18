# File: src/train_xgb.py
#!/usr/bin/env python
# coding: utf-8

# Imports & logging setup
import argparse                    # CLI parsing
import json                        # parse JSON params
import logging                     # logging messages
from typing import Dict  # type hints
import pandas as pd                # DataFrame handling
import joblib                      # model serialization
from sklearn.metrics import classification_report, roc_auc_score    # evaluation metrics
from sklearn.model_selection import train_test_split    # train/test split
import xgboost as xgb              # XGBoost
from sklearn.preprocessing import StandardScaler   # for scaling features         

# Configure logger
type_check_fmt = "%(asctime)s %(levelname)s %(message)s"    # format for type checking          
logging.basicConfig(format=type_check_fmt, level=logging.INFO)  # set logging level to INFO
logger = logging.getLogger(__name__)                        

# Core training function
def train_xgb(
    X: pd.DataFrame,    # Features as DataFrame         
    y: pd.Series,   # Target labels as Series
    xgb_params: Dict    # XGBoost parameters as dict    
) -> xgb.XGBClassifier:                                                                 
    """
    Train an XGBoost classifier with given parameters on provided data.

    Returns:
        Trained XGBClassifier.
    """
    logger.info("Initializing XGBClassifier with params: %s", xgb_params)     # Log parameters           
    model = xgb.XGBClassifier(**xgb_params, use_label_encoder=False, eval_metric='logloss')   # Initialize model with params                      
    model.fit(X, y)     # Fit model to training data
    logger.info("Training complete on %d samples.", X.shape[0])   # Log number of samples trained on
    return model

# CLI / orchestration with train/test split
def main():
    parser = argparse.ArgumentParser(   
        description="Train, evaluate, and save an XGBoost classifier using a combined feature+label CSV." # Description of the script
    )
    parser.add_argument(
        "--input-csv", required=True,
        help="CSV containing feature columns plus a 'label' column." # Input CSV path
    )
    parser.add_argument(
        "--xgb-params", default="{}",
        help="JSON string of XGB params, e.g., '{\"n_estimators\":100}'."   # XGBoost parameters as JSON string
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Proportion of data to reserve for test (0-1)."    # Proportion of data for test set (default 20%)
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed for train-test split and model."  # Random seed for reproducibility (default 42)
    )
    parser.add_argument(
        "--model-out", required=True,
        help="Path to save trained XGB model (joblib .pkl)."    # Output path for trained model
    )
    parser.add_argument(
        "--scaler-out", default=None,
        help="Path to save fitted scaler (joblib .pkl)."    # Output path for fitted scaler (optional)
    )
    parser.add_argument(
        "--report-out", dest="report_out", default=None,
        help="Path to save classification report on test set (txt)."    # Output path for classification report (optional)
    )
    args = parser.parse_args()  # Parse command-line arguments

    # Load combined data
    df = pd.read_csv(args.input_csv, index_col=0)       # Load CSV into DataFrame           
    if 'label' not in df.columns:
        logger.error("Input CSV must contain a 'label' column.")    # Ensure 'label' column exists
        raise SystemExit(1) # Ensure 'label' column exists

    # Split features and target
    y = df['label'] # Target labels 
    X = df.drop(columns='label')    # Features DataFrame    

    # Parse XGB params
    try:
        xgb_params = json.loads(args.xgb_params)        # Parse JSON string into dict
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON for --xgb-params: %s", e)    # Handle JSON parsing errors
        raise   

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,   # Features and target split
        test_size=args.test_size,       # Proportion for test set           
        random_state=args.random_state,     # Random seed for reproducibility        
        stratify=y  # Stratified split to maintain label distribution                   
    )
    
    # Fit and save scaler
    if args.scaler_out: # If scaler output path is provided
        scaler = StandardScaler()   # Initialize scaler
        scaler.fit(X_train) # Fit scaler on training data
        joblib.dump(scaler, args.scaler_out)    # Save fitted scaler to specified path
        logger.info("Saved scaler to %s", args.scaler_out)      

    logger.info("Data split: %d train, %d test samples.", X_train.shape[0], X_test.shape[0])                

    # Train model
    model = train_xgb(X_train, y_train, xgb_params)   # Train XGBoost model with provided parameters

    # Evaluate on test set
    y_pred = model.predict(X_test)  # Predict labels on test set
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None    # Get predicted probabilities if available               
    report = classification_report(y_test, y_pred)  # Generate classification report on test set
    logger.info("Test Classification Report:\n%s", report)  
    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)    # Calculate ROC AUC score if probabilities are available
        logger.info("Test ROC AUC Score: %.4f", auc)   # Log AUC score
    else:
        auc = None  # Set AUC to None if probabilities are not available

    # Save report if requested
    if args.report_out:
        with open(args.report_out, 'w') as f:   # Open report file for writing
            f.write("Classification Report on Test Set:\n" + report)    # Write classification report to file
            if auc is not None:   # If AUC is available
                f.write(f"\nROC AUC: {auc:.4f}\n")  # Write AUC score if available
        logger.info("Saved test report to %s", args.report_out)

    # Save the trained model
    joblib.dump(model, args.model_out)  # Save model to specified path
    logger.info("Saved XGB model to %s", args.model_out)

if __name__ == "__main__":
    main()
