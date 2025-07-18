# File: src/explain_shap.py
#!/usr/bin/env python
# coding: utf-8

# Imports
import os    # Ensure this script can run from any directory               
import argparse # For command-line argument parsing     
import logging  # For logging messages  
import pandas as pd # For data manipulation         
import numpy as np  # For numerical operations          
import matplotlib.pyplot as plt # For plotting SHAP values          
import shap # For SHAP explanations     
import joblib   # For loading saved models and scalers  
from tensorflow import keras    # For loading DNN models        
from sklearn.model_selection import train_test_split    # For splitting data into train/test sets           

# Enable interactive JS plots
shap.initjs()

# Configure root logger
logging.basicConfig(
    level=logging.INFO, #   Set logging level to INFO       
    format="%(asctime)s %(levelname)s %(message)s" #   Set log message format
)
logger = logging.getLogger(__name__) #  Create a logger for this module     

def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(
        description="Compute and save SHAP explanations for RF, XGB, and DNN models"
    )
    parser.add_argument("--input-csv",    required=True,
                        help="Path to input CSV with features + 'label'") # Path to input CSV file containing features and labels       
    parser.add_argument("--models-dir",   required=True,
                        help="Directory containing trained model files") # Directory where trained model files are stored
    parser.add_argument("--scalers-dir",  required=True,
                        help="Directory containing fitted scaler files") # Directory where fitted scaler files are stored   
    parser.add_argument("--figures-dir",  required=True,
                        help="Directory where SHAP summary plots will be saved") # Directory where SHAP summary plots will be saved
    parser.add_argument("--test-size",    type=float, default=0.2,
                        help="Proportion of data to hold out for testing") # Proportion of data to hold out for testing (default: 0.2)
    parser.add_argument("--random-state", type=int,   default=42,
                        help="Seed for reproducibility") # Random seed for reproducibility (default: 42)
    parser.add_argument("--show-plot",    action="store_true",
                        help="If set, display each plot interactively") # If set, display each SHAP summary plot interactively
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.figures_dir, exist_ok=True) # Create figures directory if it doesn't exist


    # Load & split
    logger.info("Loading data from %s", args.input_csv)   # Log the input CSV file path
    df = pd.read_csv(args.input_csv) # Load the input CSV file into a DataFrame
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]   # Drop any columns that start with "Unnamed" to clean up the DataFrame

    # Check for 'label' column
    if "label" not in df.columns:
        logger.error("Input CSV must contain a 'label' column.")
        raise SystemExit(1)

    # pull out X_df (DataFrame) and y (np.array)
    X_df = df.drop(columns=["label"]) # Drop the 'label' column to get features
    y    = df["label"].values  # Extract the 'label' column as a NumPy array

    # Check for NaN values
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y,    # Split the DataFrame into training and testing sets
        stratify=y, # Stratify by the 'label' column to maintain class distribution
        test_size=args.test_size, # Use the specified test size
        random_state=args.random_state # Use the specified random state for reproducibility
    )

    # Load models & scalers
    logger.info("Loading models and scalers from %s / %s",
                args.models_dir, args.scalers_dir)

    # RF
    rf_model   = joblib.load(os.path.join(args.models_dir,  "rf_model.pkl")) # Load the Random Forest model
    scaler_rf  = joblib.load(os.path.join(args.scalers_dir, "scaler_rf.pkl")) # Load the scaler for the Random Forest model

    # XGB
    xgb_model  = joblib.load(os.path.join(args.models_dir,  "xgb_model.pkl")) # Load the XGBoost model
    scaler_xgb = joblib.load(os.path.join(args.scalers_dir, "scaler_xgb.pkl")) # Load the scaler for the XGBoost model

    # DNN
    dnn_model  = keras.models.load_model(
        os.path.join(args.models_dir,  "dnn_model.keras") # Load the DNN model
    )
    scaler_dnn = joblib.load(os.path.join(args.scalers_dir, "scaler_dnn.pkl")) # Load the scaler for the DNN model

    # Check that scalers match the input features
    feat_dnn     = scaler_dnn.feature_names_in_ # Get the feature names expected by the DNN scaler
    X_train_dnn  = scaler_dnn.transform(X_train_df[feat_dnn]) # Scale the training features for the DNN model
    X_test_dnn   = scaler_dnn.transform(X_test_df[feat_dnn]) # Scale the testing features for the DNN model

    # Log expected vs provided features
    feat_dnn = scaler_dnn.feature_names_in_ # Get the feature names expected by the DNN scaler
    logger.info("DNN scaler expects: %s", list(feat_dnn)) # Log the feature names expected by the DNN scaler
    logger.info("CSV provides      : %s", list(X_train_df.columns)) # Log the feature names provided in the CSV file

    logger.info("Scaling feature sets for SHAP...")
    # RF
    feat_rf       = scaler_rf.feature_names_in_ # Get the feature names expected by the RF scaler
    X_train_rf    = scaler_rf .transform(X_train_df[feat_rf]) # Scale the training features for the RF model
    X_test_rf     = scaler_rf .transform(X_test_df[feat_rf]) # Scale the testing features for the RF model

    # XGB
    feat_xgb      = scaler_xgb.feature_names_in_ # Get the feature names expected by the XGB scaler
    X_train_xgb   = scaler_xgb.transform(X_train_df[feat_xgb]) # Scale the training features for the XGB model
    X_test_xgb    = scaler_xgb.transform(X_test_df[feat_xgb]) # Scale the testing features for the XGB model

    # DNN
    feat_dnn     = scaler_dnn.feature_names_in_ # Get the feature names expected by the DNN scaler
    X_train_dnn  = scaler_dnn.transform(X_train_df[feat_dnn]) # Scale the training features for the DNN model
    X_test_dnn   = scaler_dnn.transform(X_test_df[feat_dnn]) # Scale the testing features for the DNN model


    # Build SHAP explainers & compute values
    feat_rf  = scaler_rf .feature_names_in_ # Get the feature names expected by the RF scaler
    feat_xgb = scaler_xgb.feature_names_in_ # Get the feature names expected by the XGB scaler
    feat_dnn = scaler_dnn.feature_names_in_ # Get the feature names expected by the DNN scaler
    # Sample a small background for speed
    np.random.seed(args.random_state)
    bg_idx         = np.random.choice(X_train_rf.shape[0], 100, replace=False) # Randomly sample 100 indices from the training set for background data
    background_rf  = X_train_rf[bg_idx]  # Background data for RF
    background_xgb = X_train_xgb[bg_idx] # Background data for XGB
    background_dnn = X_train_dnn[bg_idx] # Background data for DNN

    # RF explainer
    logger.info("Building SHAP explainer for Random Forest")
    bg_rf_df     = pd.DataFrame(background_rf, columns=feat_rf) # Create a DataFrame for the background data for RF
    def rf_pos(data):
        return rf_model.predict_proba(data)[:, 1] # Define a function to get the positive class probabilities from the RF model
    explainer_rf = shap.Explainer(rf_pos, bg_rf_df) # Create a SHAP explainer for the RF model using the background data
    shap_vals_rf = explainer_rf(X_test_rf).values # Compute SHAP values for the RF model using the test set

    # XGB explainer
    logger.info("Building SHAP explainer for XGBoost")
    bg_xgb_df     = pd.DataFrame(background_xgb, columns=feat_xgb) # Create a DataFrame for the background data for XGB
    explainer_xgb = shap.TreeExplainer(xgb_model, data=bg_xgb_df) # Create a SHAP TreeExplainer for the XGB model using the background data
    shap_vals_xgb = explainer_xgb.shap_values(X_test_xgb) # Compute SHAP values for the XGB model using the test set

    # DNN explainer
    logger.info("Building SHAP explainer for DNN")
    bg_dnn_df     = pd.DataFrame(background_dnn, columns=feat_dnn)    # Create a DataFrame for the background data for DNN         
    X_eval_df     = pd.DataFrame(X_test_dnn[:200], columns=feat_dnn)  # Create a DataFrame for the evaluation data (first 200 samples of the test set)
    explainer_dnn = shap.Explainer(dnn_model, bg_dnn_df) # Create a SHAP explainer for the DNN model using the background data
    shap_exp_dnn  = explainer_dnn(X_eval_df) # Compute SHAP values for the DNN model using the evaluation data

    # Generate & save SHAP summary plots
    # RF
    logger.info("Plotting RF SHAP summary")
    rf_df = pd.DataFrame(X_test_rf, columns=feat_rf) # Create a DataFrame for the test set features for RF
    shap.summary_plot(shap_vals_rf, rf_df, show=False) # Generate a SHAP summary plot for the RF model
    plt.title("RF SHAP Summary — P(class = 1)") # Set the title for the RF SHAP summary plot
    plt.tight_layout() # Adjust layout to prevent overlap
    plt.savefig(os.path.join(args.figures_dir, "rf_shap_summary.png")) # Save the RF SHAP summary plot
    plt.clf() # Clear the current figure to free memory

    # XGB
    logger.info("Plotting XGB SHAP summary")
    xgb_df = pd.DataFrame(X_test_xgb, columns=feat_xgb) # Create a DataFrame for the test set features for XGB
    shap.summary_plot(shap_vals_xgb, xgb_df, show=False) # Generate a SHAP summary plot for the XGB model
    plt.title("XGBoost SHAP Summary — P(class = 1)") # Set the title for the XGB SHAP summary plot
    plt.tight_layout() # Adjust layout to prevent overlap
    plt.savefig(os.path.join(args.figures_dir, "xgb_shap_summary.png")) # Save the XGB SHAP summary plot
    plt.clf() # Clear the current figure to free memory

    # DNN
    logger.info("Plotting DNN SHAP summary")
    vals = shap_exp_dnn.values # Extract SHAP values from the DNN explainer
    shap.summary_plot(vals, X_eval_df, show=False) # Generate a SHAP summary plot for the DNN model
    plt.title("DNN SHAP Summary — P(class = 1)") # Set the title for the DNN SHAP summary plot
    plt.tight_layout() # Adjust layout to prevent overlap
    plt.savefig(os.path.join(args.figures_dir, "dnn_shap_summary.png")) # Save the DNN SHAP summary plot

if __name__ == "__main__":
    main()