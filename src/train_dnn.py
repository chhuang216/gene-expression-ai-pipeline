# File: src/train_dnn.py
#!/usr/bin/env python
# coding: utf-8

import argparse # for command-line argument parsing
import logging # for logging
from sklearn.preprocessing import StandardScaler # for feature scaling
import os # for file and directory operations
import pandas as pd # for data manipulation
import joblib # for saving and loading models and scalers
from sklearn.model_selection import train_test_split # for splitting data into train and test sets
from sklearn.metrics import ( 
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix # for model evaluation metrics
)
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for enhanced plotting
from tensorflow.keras.models import Sequential # for building sequential models
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # for adding layers to the model
from tensorflow.keras.optimizers import Adam # for optimization
from keras_tuner.tuners import RandomSearch # for hyperparameter tuning
from tensorflow.keras import callbacks # for callbacks like early stopping
import shutil # for file operations like removing directories

# Logger setup
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# Cleanup old KerasTuner logs
tuner_dir = os.path.join("ktuner_logs", "dnn_finetune") # directory for KerasTuner logs
if os.path.isdir(tuner_dir):
    logger.info("Removing old tuner logs at %s", tuner_dir)
    try:
        # ignore_errors=True silences Windows permission issues
        shutil.rmtree(tuner_dir, ignore_errors=True) # remove the directory and its contents
    except Exception as e:
        logger.warning("Could not fully remove %s: %s", tuner_dir, e)

# DNN model builder for KerasTuner
def build_model(hp):
    model = Sequential() # initialize a sequential model
    num_layers = hp.Int('num_layers', 1, 4) # number of hidden layers to add, between 1 and 4

    for i in range(num_layers):
        model.add(Dense( # add a dense layer
            units=hp.Int(f'units_{i}', 32, 512, step=32), # number of units in the layer
            activation=hp.Choice(f'activation_{i}', ['relu', 'tanh', 'sigmoid']) # activation function
        ))
        if hp.Boolean(f'batchnorm_{i}'): # optional batch normalization
            model.add(BatchNormalization()) # add batch normalization layer
        if hp.Boolean(f'dropout_{i}'): # optional dropout
            model.add(Dropout(hp.Float(f'drop_rate_{i}', 0.1, 0.5, step=0.1))) # add dropout layer

    model.add(Dense(1, activation='sigmoid')) # output layer for binary classification

    model.compile( # compile the model
        optimizer=Adam(hp.Float('lr', 1e-5, 1e-2, sampling='log')), # learning rate
        loss='binary_crossentropy', # loss function for binary classification
        metrics=['accuracy'] # metrics to monitor during training
    )
    return model

# Plot functions
def plot_loss(history, out_path=None):
    plt.figure(figsize=(8, 6)) # create a figure for loss curves
    plt.plot(history.history['loss'], label='loss')     # training loss
    plt.plot(history.history['val_loss'], label='val_loss') # validation loss
    plt.xlabel('Epoch') # x-axis label
    plt.ylabel('Loss') # y-axis label
    plt.legend() # add legend
    plt.tight_layout() # adjust layout
    if out_path:
        plt.savefig(out_path) # save the plot to a file
        logger.info("Loss curves saved to %s", out_path)

def plot_confusion_matrix(y_true, y_pred, out_path=None):
    cm = confusion_matrix(y_true, y_pred) # compute confusion matrix
    plt.figure(figsize=(6, 5)) # create a figure for confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') # heatmap for confusion matrix
    plt.ylabel('Actual') # y-axis label
    plt.xlabel('Predicted') # x-axis label
    plt.tight_layout() # adjust layout
    if out_path:
        plt.savefig(out_path) # save the plot to a file
        logger.info("Confusion matrix saved to %s", out_path)

# Main training pipeline
def main():
    parser = argparse.ArgumentParser(description="Train a DNN model with hyperparameter tuning.") # argument parser for command-line options
    parser.add_argument("--input-csv", required=True, help="Input CSV file with features and label.") # input CSV file with features and labels
    parser.add_argument("--model-out", required=True, help="Output path for trained DNN model.") # output path for the trained DNN model
    parser.add_argument("--report-out", required=True, help="Path to save classification report.") # path to save the classification report
    parser.add_argument("--loss-plot-out", default=None, help="Path to save loss curves plot.") # path to save the loss curves plot
    parser.add_argument("--matrix-plot-out", default=None, help="Path to save confusion matrix plot.") # path to save the confusion matrix plot
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion for test set.") # proportion of data to use for the test set
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility.") # random state for reproducibility
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum number of iteration.") # maximum number of hyperparameter combinations to try
    parser.add_argument("--epochs", type=int, default=1000, help="Maximum number of epochs.") # maximum number of epochs for training
    parser.add_argument("--patient", type=int, default=5, help="Maximum number of epochs.") # patience for early stopping
    parser.add_argument("--scaler-out", default=None, help="Path to save fitted scaler.") # path to save the fitted scaler
    args = parser.parse_args() # parse command-line arguments

    # Load data
    df = pd.read_csv(args.input_csv) # read the input CSV file
    X  = df.drop('label', axis=1) # features (all columns except 'label')
    y  = df['label'] # labels (the 'label' column)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split( 
        X, y,  # split data into training and test sets
        test_size=args.test_size,  # proportion of data to use for the test set
        random_state=args.random_state,  # for reproducibility
        stratify=y # stratify to maintain class distribution
    )
    
    scaler = StandardScaler() # initialize a standard scaler
    X_train_scaled = scaler.fit_transform(X_train) # fit and transform the training data
    X_test_scaled = scaler.transform(X_test) # transform the test data using the fitted scaler

    # Replace original X_train/X_test with scaled versions
    X_train = X_train_scaled 
    X_test = X_test_scaled 

    # Save scaler
    if args.scaler_out:
        os.makedirs(os.path.dirname(args.scaler_out), exist_ok=True) # ensure the directory exists
        joblib.dump(scaler, args.scaler_out) # save the fitted scaler to a file
        logger.info("Scaler saved to %s", args.scaler_out) 

    # Hyperparameter tuning
    tuner = RandomSearch(
        build_model, # function that builds models
        objective='val_accuracy', # objective to optimize (validation accuracy)
        max_trials=args.max_iter, # maximum number of hyperparameter combinations to try
        executions_per_trial=3, # average performance over 3 runs each
        directory='ktuner_logs', # directory to save logs
        project_name='dnn_finetune' # project name for KerasTuner
    )
    # Early stopping to prevent overfitting
    es = callbacks.EarlyStopping( 
        monitor="val_loss",  # monitor validation loss
        patience=args.patient,  # number of epochs with no improvement after which training will be stopped
        restore_best_weights=True # restore the best weights after early stopping
    )

    # Compute class weights to address imbalance
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum() # count negative and positive samples
    class_weight = {0: 1, 1: neg / pos} # compute class weights to balance the classes

    # Launch search
    tuner.search(
        X_train, y_train,           # training data and labels
        epochs=args.epochs,         # maximum epochs per trial
        validation_split=0.2,       # use 20% of train for validation
        callbacks=[es],             # apply early stopping
        class_weight=class_weight   # apply class weights to handle imbalance 
    )
    # Display the search space for verification
    tuner.search_space_summary()    # display the search space summary
    

    best_model = tuner.get_best_models(num_models=1)[0] # get the best model from the tuner
    best_hp    = tuner.get_best_hyperparameters()[0].values # get the best hyperparameters from the tuner
    print("Best hyperparameters:", best_hp) 

    # Train best model
    history = best_model.fit(
        X_train, y_train, # train the best model on the training data
        validation_split=0.2, # use 20% of training data for validation
        epochs=args.epochs, # maximum number of epochs
        callbacks=[es], # apply early stopping
        class_weight=class_weight, # apply class weights to handle imbalance
        verbose=1 # verbose output during training
    )

    # Evaluation
    y_probs = best_model.predict(X_test).ravel().astype(float) # get predicted probabilities for the test set
    y_pred = (best_model.predict(X_test) >= 0.5).astype(int) # convert probabilities to binary predictions
    
    report = classification_report(y_test, y_pred) # generate classification report
    
    # Compute core metrics
    acc  = accuracy_score(y_test, y_pred) # compute accuracy
    prec = precision_score(y_test, y_pred) # compute precision
    rec  = recall_score(y_test, y_pred) # compute recall
    f1   = f1_score(y_test, y_pred) # compute F1 score
    auc  = roc_auc_score(y_test, y_probs) # compute ROC AUC score
    
    # Log metrics
    logger.info("DNN Metrics Report:\n%s", report) 
    logger.info("Accuracy: %.4f", acc)
    logger.info("Precision: %.4f", prec)
    logger.info("Recall: %.4f", rec)
    logger.info("F1 Score: %.4f", f1)
    logger.info("ROC AUC: %.4f", auc)

    # Save model
    best_model.save(args.model_out) # save the trained model to a file
    logger.info("Model saved to %s", args.model_out)

    # Save report
    with open(args.report_out, 'w') as f:
        f.write(report) # save the classification report to a file
        f.write(f"\nROC AUC: {auc:.4f}\n") 
    logger.info("Report saved to %s", args.report_out)

    # Plotting
    if args.loss_plot_out:
        plot_loss(history, out_path=args.loss_plot_out) # plot training vs validation loss
    if args.matrix_plot_out:
        plot_confusion_matrix(y_test, y_pred, out_path=args.matrix_plot_out) # plot confusion matrix

if __name__ == '__main__':
    main()
