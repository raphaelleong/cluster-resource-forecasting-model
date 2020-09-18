# Overview

# Cluster-data
Public github repo: https://github.com/google/cluster-data
overview of the Google cluster trace which provides details and instructions on how to obtain the trace.
Downloading compressed data 'clusterdata-2011-2' is required for files in this repository. (approx. 41 GB)
clusterdata-2011-2 is not included in this compressed repo due to CATe file size limit. 

# Python analyzer implementation
python_analyzer.py - parses and analyzes large chunks of CSV data provided from clusterdata-2011-2.
Samples and processes the data to store in compiled_data directories for training the forecasting model.

# Compiled_data directories
These are the CSV datasets used for training the forecasting model:

compiled_data - 24 hour time period, collected samples with 5 min sampling rate

compiled_data_inc - 7 day time period, collected samples with 5 min sampling rate

compiled_data_ssampling - 24 hour time period, collected samples with 1 min sampling rate

compiled_data_ssampling_test - 24 hour time period, collected samples with 1 min sampling rate
 (sampled from different starting point for testing the model)

# Forecasting model implementation
preprocessing.py - loads data from compiled_data folders and modifies data format to fit LSTM input

forecasting_model.py - Main convolutional LSTM neural network model training and evaluation

online_forecasting_model_extension.py - exploring different approaches and hyperparameters to training the model 
