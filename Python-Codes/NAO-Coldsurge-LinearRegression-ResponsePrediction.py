# -*- coding: utf-8 -*-
"""
Created on 11/1/2024

@author: yhw
"""

### goals: build a linear regression model with intervals for prediction
### using t-distribution for intervals

### Load libraries needed
import numpy as np
import pandas as pd
import statsmodels.api as sm  # update package to avoid errors
import matplotlib.pyplot as plt



### Load data from text files
# winter-mean data
nao = np.loadtxt(r'D:\yhw\Weather-Climate-Collaboration\NAO-related\nao_1940_2022.txt')
csf = np.loadtxt(r'D:\yhw\Weather-Climate-Collaboration\NAO-related\csf_1940_2022.txt')



### Process data
# Convert to pandas series for easier manipulation
nao_series = pd.Series(nao)
csf_series = pd.Series(csf)

# Calculate 7-year running mean to filter out interannual variability
nao_smooth = nao_series.rolling(window=7, center=True).mean()
csf_smooth = csf_series.rolling(window=7, center=True).mean()



### Plot time series
# Define years from 1940 to 2022
years = np.arange(1940, 1940 + len(nao_smooth))

plt.figure(figsize=(12, 6))

# Plot time series for NAO
plt.subplot(2, 1, 1)
plt.plot(years, nao_series, label='Original NAO', color='blue', alpha=0.7)
plt.plot(years, nao_smooth, label='7-Year Running Mean NAO', color='orange', linewidth=2)
plt.title('Time Series of NAO with 7-Year Running Mean')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()

# Plot time series for CSF
plt.subplot(2, 1, 2)
plt.plot(years, csf_series, label='Original CSF', color='green', alpha=0.7)
plt.plot(years, csf_smooth, label='7-Year Running Mean CSF', color='red', linewidth=2)
plt.title('Time Series of CSF with 7-Year Running Mean')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
#plt.show()

# Save the first plot to a file
plt.savefig(r'D:\yhw\Weather-Climate-Collaboration\NAO-related\NAO-CSF-7yrmean-ts.png', format='png', dpi=300)  # Specify your path and desired format
plt.close()  # Close the figure to free up memory



### Build linear regression
# Remove NaN values created by the rolling mean
nao_smooth = nao_smooth.dropna().values
csf_smooth = csf_smooth.dropna().values

# nao_smooth is the independent variable (x), and csf_smooth is the dependent variable (y)
X = sm.add_constant(nao_smooth)  # Add a constant for the intercept

# Fit a linear regression model
model = sm.OLS(csf_smooth, X).fit()
slope = model.params[1]
intercept = model.params[0]

# Calculate the correlation coefficient
correlation = np.corrcoef(nao_smooth, csf_smooth)[0, 1]

# Define prediction points (e.g., same points as x or a new range)
pred_x = np.linspace(min(nao_smooth), max(nao_smooth), 100)
X_pred = sm.add_constant(pred_x)



### Interval estimates
# Get predictions with confidence intervals and prediction intervals
predictions = model.get_prediction(X_pred)
summary_frame = predictions.summary_frame(alpha=0.1)  # 90% CI, adjust alpha as needed

# Extract values for plotting
mean_pred = summary_frame['mean']
conf_int_lower = summary_frame['mean_ci_lower']  # 5% CI for mean response
conf_int_upper = summary_frame['mean_ci_upper']  # 95% CI for mean response
pred_int_lower = summary_frame['obs_ci_lower']   # 5% PI for individual prediction
pred_int_upper = summary_frame['obs_ci_upper']   # 95% PI for individual prediction

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(nao_smooth, csf_smooth, color='black', label='Data')  # Data points in black
plt.plot(pred_x, mean_pred, color='black', label='Regression Line')  # Regression line in black
plt.fill_between(pred_x, conf_int_lower, conf_int_upper, color='blue', alpha=0.2, label='Confidence Interval (Mean Response)')  # Confidence interval for mean in blue
plt.fill_between(pred_x, pred_int_lower, pred_int_upper, color='red', alpha=0.2, label='Prediction Interval (Individual Prediction)')  # Prediction interval in red
plt.xlabel('NAO (7-year running mean)')
plt.ylabel('CSF (7-year running mean)')
plt.legend()
plt.title('Regression with Confidence and Prediction Intervals')

# Add the regression equation and correlation coefficient to the plot
equation_text = f'Regression line: y = {slope:.2f}x + {intercept:.2f}\nCorrelation: r = {correlation:.2f}'
plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
#plt.show()

# Save the first plot to a file
plt.savefig(r'D:\yhw\Weather-Climate-Collaboration\NAO-related\NAO-CSF-7yrmean-scatter.png', format='png', dpi=300)  # Specify your path and desired format
plt.close()  # Close the figure to free up memory
