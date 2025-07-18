# File: src/feature_selection.py
#!/usr/bin/env python
# coding: utf-8

# Imports & logging setup
import argparse                # for command-line argument parsing
import json                    # for parsing JSON strings
import logging                 # for logging messages
from typing import Dict, Tuple, Optional  # for type hints
import pandas as pd            # for DataFrame handling
import matplotlib.pyplot as plt # for plotting
from sklearn.ensemble import RandomForestClassifier  # for feature selection

# Configure logger
type_check_fmt = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(format=type_check_fmt, level=logging.INFO) # set logging format and level
logger = logging.getLogger(__name__)


# Core feature-selection function
def select_top_features(
    X: pd.DataFrame, # Feature DataFrame
    y: pd.Series, # Target Series
    k: int, # Number of top features to select
    rf_params: Dict = None # Optional RF parameters
) -> Tuple[pd.DataFrame, pd.Series]: # Returns selected features and importances
    """
    Train a RandomForestClassifier to compute feature importances,
    then select top-k features.

    Returns:
        X_sel: DataFrame of top-k features.
        importances: Series of all feature importances.
    """
    rf_params = rf_params or {"n_estimators": 100, "random_state": 42} # Default RF params
    logger.info("Training RF for feature importances with params: %s", rf_params)
    rf = RandomForestClassifier(**rf_params) # Initialize RF with provided params
    rf.fit(X, y) # Fit RF to data

    importances = pd.Series(rf.feature_importances_, index=X.columns) # Get feature importances
    top_feats = importances.nlargest(k).index.tolist() # Select top-k features
    logger.info("Selected top %d features", k)
    return X[top_feats], importances # Return selected features and all importances

# Plotting helper
def plot_importances(
    importances: pd.Series, # Series of feature importances
    k: int, # Number of top features to plot
    show: bool = False, # If True, display the plot
    out_path: Optional[str] = None # If provided, save the plot to this path
) -> None:
    """
    Plot and/or save top-k feature importances.

    Args:
        importances: Series of feature importances.
        k: Number of top features to include.
        show: If True, display the plot.
        out_path: If provided, save the plot to this path.
    """
    top_imp = importances.nlargest(k) # Get top-k importances
    plt.figure(figsize=(10, 6)) # Set figure size
    top_imp.plot(kind='bar') # Create bar plot of importances
    plt.title(f"Top {k} Feature Importances") # Set plot title
    plt.xlabel("Feature") # Set x-axis label
    plt.ylabel("Importance") # Set y-axis label
    plt.tight_layout() # Adjust layout for better fit
    if out_path:
        plt.savefig(out_path) # Save plot if out_path is provided
        logger.info("Saved importance plot to %s", out_path)
    if show:
        plt.show()

# CLI entry point and orchestration
def main():
    parser = argparse.ArgumentParser(
        description="Select top-k features and optionally plot/save importances."
    )
    parser.add_argument(
        "--input-csv", required=True, # Input CSV with features and labels
        help="CSV with feature columns plus 'label' column."
    )
    parser.add_argument(
        "--k", type=int, required=True, # Number of top features to select
        help="Number of top features to select and plot."
    )
    parser.add_argument(
        "--rf-params", default="{}", # JSON string of RF parameters
        help="JSON string of RF params, e.g., '{\"n_estimators\":200}'."
    )
    parser.add_argument(
        "--out-csv", required=True, # Output CSV for selected features
        help="Path to write selected features CSV."
    )
    parser.add_argument(
        "--plot", action="store_true", # If set, display the feature importances plot
        help="Display the feature importances plot."
    )
    parser.add_argument(
        "--plot-out", default=None, # If provided, save the plot to this path
        help="Path to save the feature importances plot (PNG)."
    )

    args = parser.parse_args()

    # Load combined data
    df = pd.read_csv(args.input_csv, index_col=0)
    if 'label' not in df.columns:
        logger.error("Input CSV must contain a 'label' column.")
        raise SystemExit(1) # Ensure 'label' column exists

    # Split X and y
    y = df['label'] # Extract labels
    X = df.drop(columns='label') # Drop label column to get features

    # Parse RF params
    try:
        rf_params = json.loads(args.rf_params) # Convert JSON string to dict
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON for --rf-params: %s", e) 
        raise

    # Feature selection
    X_sel, importances = select_top_features(X, y, args.k, rf_params) # Select top-k features

    # Plot/save importances if requested
    if args.plot or args.plot_out:
        plot_importances(importances, args.k, show=args.plot, out_path=args.plot_out) # Plot and save importances if requested
    
    X_sel['label'] = y.values   # Ensure labels are included in the output

    # Save selected features + labels
    X_sel.to_csv(args.out_csv, index=False) # Save selected features to CSV
    logger.info("Filtered features + labels saved to %s", args.out_csv)

if __name__ == "__main__":
    main()