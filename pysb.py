import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import statsmodels.api as sm

def pysb(y, min_pct=0.15):
    """
    Performs a Supremum Wald Test for unknown break dates in a single time series and visualizes 
    the original time series with the breakpoint, preserving the datetime index.

    Parameters:
    - y: The time series (as a pandas Series with datetime index).
    - min_pct: Minimum percentage of observations before and after a breakpoint to be considered valid.

    Returns:
    - sup_wald_stat: The supremum of the Wald test statistics.
    - breakpoint: The datetime index of the most significant break date.
    - p_value: The p-value of the Supremum Wald statistic.
    """

    # Ensure y is a pandas Series with a datetime index
    if not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series with a datetime index")

    # Store the datetime index for plotting
    dates = y.index

    # Convert the values of y to a numpy array for processing
    y_values = y.values

    # Length of the time series
    n = len(y_values)

    # Define the minimum and maximum possible breakpoints
    min_obs = int(np.floor(n * min_pct))
    max_obs = n - min_obs

    # Initialize lists to store the Wald statistics and corresponding breakpoints
    wald_stats = []
    breakpoints = []

    for b in range(min_obs, max_obs):
        # Fit the models before and after the breakpoint (mean changes model)
        model1 = sm.OLS(y_values[:b], np.ones(b)).fit()
        model2 = sm.OLS(y_values[b:], np.ones(n - b)).fit()

        # Get the residual sum of squares for both models
        RSS1 = np.sum(model1.resid ** 2)
        RSS2 = np.sum(model2.resid ** 2)

        # Combine the residuals to get the total RSS
        RSS_total = RSS1 + RSS2

        # Estimate the unrestricted model (whole period)
        model_full = sm.OLS(y_values, np.ones(n)).fit()
        RSS_full = np.sum(model_full.resid ** 2)

        # Compute the Wald statistic (for structural change in mean)
        k = 1  # number of parameters (just the mean in this case)
        wald_stat = (RSS_full - RSS_total) / k / (RSS_total / (n - 2 * k))

        # Append Wald statistic and breakpoint
        wald_stats.append(wald_stat)
        breakpoints.append(b)

    # Find the supremum Wald statistic and the corresponding breakpoint
    sup_wald_stat = max(wald_stats)
    breakpoint_index = breakpoints[wald_stats.index(sup_wald_stat)]
    breakpoint = dates[breakpoint_index]  # Get the actual datetime corresponding to the breakpoint

    # Calculate p-value using chi-square distribution
    p_value = 1 - chi2.cdf(sup_wald_stat, df=k)

    # Visualization of the original time series with the detected breakpoint
    plt.figure(figsize=(10, 6))
    plt.plot(dates, y_values, label='Time Series', color='blue', marker='o')
    plt.axvline(breakpoint, color='red', linestyle='--', label=f'Breakpoint: {breakpoint.date()}')
    plt.title('Time Series with Supremum Wald Test Breakpoint')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    return sup_wald_stat, breakpoint, p_value
