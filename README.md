# Downscaling-Climate

# Task I: Downscaling Precipitation Data in Taiwan with the CNN-Gamma Model

This repository documents the workflow for downscaling precipitation data in Taiwan using the CNN-Gamma model. The process is divided into three main steps, as outlined below:

1. Identify a Reasonable Hyperparameter Configuration
Script: Hyperparameter-Tuning-CNN-Gamma-Nonlinear-Taiwan-LearningCurve-MSEcomp.ipynb
Description: This step focuses on tuning hyperparameters for the CNN-Gamma model. The model's performance is evaluated by comparing the training and validation learning curves using the Mean Squared Error (MSE).
2. Train a CNN-Gamma Model for Further Evaluation
Script: Training_Baseline_Models-Rampal2Taiwan.ipynb
Description: This script builds a baseline model using data specific to Taiwan. It is adapted from the original scripts provided by Mr. Rampal to serve as a foundation for subsequent steps.
3. Identify Optimal Combinations of Atmospheric Variable Inputs
Script: InputVariable-Model-Evaluation-CNNGamma-Taiwan.ipynb
Description: This step evaluates various combinations of atmospheric variables to train the CNN-Gamma model (using the model from Step 2) and predict precipitation. Performance is assessed using several metrics, with considerations for potential improvements:
Focus on wet days (e.g., precipitation > 1 mm) to refine evaluation metrics.
Assess the impact of input variable combinations on the model’s ability to predict precipitation effectively.


# Other Notebooks
hyperparameter_tuning_simpledense_text.ipynb: Example of tuning hyperparameters for a simple dense model

Hyperparameter_Tuning_CNN_Gamma_Nonlinear_text.ipynb: Example of hyperparameter tuning for a CNN-Gamma model


# Python-Codes
NAO-Coldsurge-LinearRegression-ResponsePrediction.py: Plot time series of NAO and Cold Surge Indices and build a linear regression model with intervals based on t-distribution

NAO-Coldsurge-LinearRegression-Bootstrapping.py: Build a linear regression model with intervals based on bootstrapping
