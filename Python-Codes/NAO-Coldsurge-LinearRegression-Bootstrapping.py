# -*- coding: utf-8 -*-
"""
Created on 11/1/2024

@author: yhw
"""

### goals: build a linear regression model with intervals for prediction
### using bootstrapping for intervals

### Load libraries needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression



### Process data
# Load data from text files
nao = np.loadtxt(r'D:\yhw\Weather-Climate-Collaboration\NAO-related\nao_1940_2022.txt')
csf = np.loadtxt(r'D:\yhw\Weather-Climate-Collaboration\NAO-related\csf_1940_2022.txt')

# Convert to pandas series for easier manipulation
nao_series = pd.Series(nao)
csf_series = pd.Series(csf)

# Calculate 7-year running means and drop NaNs
nao_smooth = nao_series.rolling(window=7, center=True).mean().dropna().values
csf_smooth = csf_series.rolling(window=7, center=True).mean().dropna().values



### Bootstrapping
# Define number of bootstrap samples
n_bootstrap_samples = 1000
intercepts = []
slopes = []

# Perform bootstrapping
for _ in range(n_bootstrap_samples):
    # Resample data
    nao_resample, csf_resample = resample(nao_smooth, csf_smooth)
    
    # Fit linear regression model
    model = LinearRegression().fit(nao_resample.reshape(-1, 1), csf_resample)
    intercepts.append(model.intercept_)
    slopes.append(model.coef_[0])

# Calculate mean intercept and slope for the final equation
mean_intercept = np.mean(intercepts)
mean_slope = np.mean(slopes)
equation_text = f'y = {mean_slope:.3f}x + {mean_intercept:.3f}'



### Plot 
# Create the scatter plot of the running-mean data points
plt.figure(figsize=(10, 6))
plt.scatter(nao_smooth, csf_smooth, color='blue', alpha=0.5, label='7-year Running Mean Data')

# Plot the regression line based on mean coefficients
x_vals = np.linspace(min(nao_smooth), max(nao_smooth), 100)
y_vals = mean_slope * x_vals + mean_intercept
plt.plot(x_vals, y_vals, color='red', label='Mean Regression Line')
plt.ylim(-5, 5)  # Set y-axis limits

# Calculate 95% confidence intervals
lower_bound = np.percentile(slopes, 2.5) * x_vals + np.percentile(intercepts, 2.5)
upper_bound = np.percentile(slopes, 97.5) * x_vals + np.percentile(intercepts, 97.5)
plt.fill_between(x_vals, lower_bound, upper_bound, color='gray', alpha=0.3, label='95% Confidence Interval')

# Add equation text at the top of the plot
plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Final plot adjustments
plt.xlabel('NAO (7-year Running Mean)')
plt.ylabel('CSF (7-year Running Mean)')
plt.title('Linear Regression with Bootstrapping on 7-year Running Mean Data')
plt.legend()
plt.show()

# Save the first plot to a file
plt.savefig(r'D:\yhw\Weather-Climate-Collaboration\NAO-related\NAO-CSF-7yrmean-scatter-bootstrapping.png', format='png', dpi=300)  # Specify your path and desired format
plt.close()  # Close the figure to free up memory
