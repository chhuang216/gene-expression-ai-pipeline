# File: src/preprocess.py
#!/usr/bin/env python
# coding: utf-8

# Imports & logging setup
import argparse # Command-line argument parsing
import gzip # For reading gzipped files
import logging # Logging setup
from io import StringIO # For reading strings as file-like objects
import matplotlib.pyplot as plt # For plotting elbow curves
import pandas as pd # For data manipulation
from kneed import KneeLocator # For finding the elbow point in variance

# Configure logger
type_check_fmt = "%(asctime)s %(levelname)s %(message)s" # Format for logging messages
logging.basicConfig(format=type_check_fmt, level=logging.INFO) # Set logging level to INFO
logger = logging.getLogger(__name__) # Create a logger for this module


# Core preprocessing functions
def load_expression_matrix(path: str) -> pd.DataFrame: 
    """
    Read GEO series_matrix.gz and return a transposed expr DataFrame.
    """
    with gzip.open(path, "rt") as f:
        lines = f.readlines() # Read all lines from the gzipped file
    start = next(i for i, L in enumerate(lines) if L.startswith("!series_matrix_table_begin")) # Find start of data section
    end = next(i for i, L in enumerate(lines) if L.startswith("!series_matrix_table_end")) # Find end of data section
    df = pd.read_csv(StringIO("".join(lines[start+1:end])), sep="\t", index_col=0) # Read the data into a DataFrame
    df.columns = df.columns.str.replace('"', '').str.replace('GSM', '') # Clean up column names
    return df.T     # Transpose the DataFrame to have samples as rows


def parse_labels(path: str) -> pd.DataFrame:
    """
    Extract relapse labels from the .soft file and return a DataFrame indexed by Sample.
    """
    labels = {} # Dictionary to hold sample labels
    current = None # Current sample being processed
    with gzip.open(path, "rt") as f: # Open the gzipped .soft file
        for line in f: # Iterate through each line
            if line.startswith("!Sample_geo_accession"): # Check for sample accession line
                current = line.strip().split()[-1].replace("GSM", "") # Extract sample ID
            if current and line.startswith("!Sample_characteristics_ch1") and "bone relapses" in line.lower(): # Check for relapse label line
                val = line.strip().split(":")[-1].strip() # Extract the label value
                if val in {"0", "1"}: # Only keep binary labels
                    labels[current] = int(val)  # Store label in dictionary
    df = pd.DataFrame.from_dict(labels, orient="index", columns=["label"]) # Convert dictionary to DataFrame
    df.index.name = "Sample" # Set index name for clarity
    return df # Return DataFrame with samples as index and labels as column


def merge_data(expr: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame: 
    """
    Keep only samples present in both and merge into one DataFrame.
    """
    common = expr.index.intersection(labels.index) # Find common samples in both DataFrames
    expr_sub = expr.loc[common] # Subset expression DataFrame to common samples
    lbl_sub = labels.loc[common] # Subset labels DataFrame to common samples
    merged = expr_sub.merge(lbl_sub, left_index=True, right_index=True) # Merge the two DataFrames on index
    return merged # Return merged DataFrame with samples as index and labels as column


def find_elbow_variance(df: pd.DataFrame, plot: bool = False) -> int: 
    """
    Compute per-gene variance, optionally plot the elbow, and return elbow index.
    """
    X = df.drop(columns="label") # Drop label column to focus on gene expression data
    vars_sorted = X.var(axis=0).sort_values(ascending=False) # Compute variance for each gene and sort in descending order
    if plot:
        plt.figure(figsize=(10, 6)) # Set figure size for the plot
        plt.plot(vars_sorted.values, label="Gene Variance") # Plot the variance values
        plt.title("Elbow Plot of Gene Variance") # Set plot title
        plt.xlabel("Ranked Genes") # Set x-axis label
        plt.ylabel("Variance") # Set y-axis label
        plt.grid(True) # Enable grid for better readability
        plt.tight_layout() # Adjust layout to fit labels
        plt.show() # Show the plot if requested
    knee = KneeLocator(range(len(vars_sorted)), vars_sorted.values, curve="convex", direction="decreasing") # Use KneeLocator to find the elbow point
    return int(knee.knee) # Return the index of the elbow point as an integer


def filter_top_genes(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """
    Return DataFrame containing only the top_k highest-variance genes plus label.
    """
    X = df.drop(columns="label") # Drop label column to focus on gene expression data
    top = X.var(axis=0).nlargest(top_k).index # Get the indices of the top_k genes with highest variance
    out = df[top].copy() # Create a new DataFrame with only the top_k genes
    out["label"] = df["label"] # Add the label column back to the DataFrame
    return out # Return the filtered DataFrame with top_k genes and labels


# CLI / orchestration
def main():
    parser = argparse.ArgumentParser(description="Preprocess GEO data and filter by variance elbow or fixed top-k.")  # Create argument parser for command-line interface
    parser.add_argument("--series-matrix", required=True, help="Path to GSE series_matrix.txt.gz") # Path to the expression matrix file
    parser.add_argument("--family-soft", required=True, help="Path to GSE family.soft.gz") # Path to the family .soft file containing labels
    parser.add_argument("--out-csv", required=True, help="Where to save filtered CSV") # Path to save the output CSV with filtered genes and labels
    parser.add_argument("--plot", action="store_true", help="Show the elbow plot") # Whether to display the elbow plot
    parser.add_argument("--plot-out", default=None, help="Path to save the elbow plot as PNG") # Path to save the elbow plot as a PNG file
    parser.add_argument("--top-k", type=int, default=None, help="Override elbow and select this many genes") # Number of top genes to keep based on variance, overrides elbow detection if specified
    args = parser.parse_args()

    logger.info("Loading expression matrix from %s", args.series_matrix)   # Log the path to the expression matrix file
    expr = load_expression_matrix(args.series_matrix)  # Load the expression matrix and transpose it
    logger.info("Parsing labels from %s", args.family_soft) # Log the path to the family .soft file
    labels = parse_labels(args.family_soft) # Parse the labels from the family .soft file

    logger.info("Merging data; samples before merge: expr=%d, labels=%d", expr.shape[0], labels.shape[0]) # Log the number of samples before merging
    merged = merge_data(expr, labels) # Merge the expression matrix and labels into a single DataFrame

    if args.top_k is not None: # Check if a specific number of top genes is provided
        k = args.top_k # Use the provided top_k value
        logger.info("Overriding elbow: using top_k=%d", k)  # Log that we are using the specified top_k value
    else:
        logger.info("Finding elbow point (plot=%s)", args.plot) # Log that we will find the elbow point
        k = find_elbow_variance(merged, plot=args.plot) # Find the elbow point in the variance of genes
        logger.info("Elbow found at k=%d", k) # Log the elbow point found

    filtered = filter_top_genes(merged, top_k=k) # Filter the DataFrame to keep only the top_k genes with highest variance plus labels
    filtered.to_csv(args.out_csv, index=False) # Save the filtered DataFrame to a CSV file
    logger.info("Saved top %d genes + labels to %s", k, args.out_csv) # Log the path where the filtered CSV is saved

    # Handle plot saving
    if args.plot_out: # Check if a path for saving the plot is provided
        # recompute variances for plotting
        X = merged.drop(columns="label") # Drop label column to focus on gene expression data
        vars_sorted = X.var(axis=0).sort_values(ascending=False).values # Compute variance for each gene and sort in descending order
        knee = KneeLocator(range(len(vars_sorted)), vars_sorted, curve="convex", direction="decreasing") # Use KneeLocator to find the elbow point again for plotting

        plt.figure(figsize=(10,6)) # Set figure size for the plot
        plt.plot(vars_sorted, label="Gene Variance") # Plot the variance values
        plt.axvline(knee.knee, color="red", linestyle="--", label=f"Elbow at {knee.knee}") # Draw a vertical line at the elbow point
        plt.legend() # Add legend to the plot
        plt.tight_layout()  # Adjust layout to fit labels
        plt.savefig(args.plot_out) # Save the plot to the specified path
        logger.info("Saved elbow plot to %s", args.plot_out) # Log the path where the plot is saved


if __name__ == "__main__": 
    main()
