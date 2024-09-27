# pySB
PySB: Supremum Wald Test for Structural Breakpoints in Time Series
This Python module performs a Supremum Wald Test to identify unknown break dates in a single time series. The test identifies structural breaks in the mean of a time series and visualizes the time series with the detected breakpoints, preserving the datetime index.

Features\

* Supremum Wald Test: Identifies breakpoints in the mean of a time series using Wald statistics.
* Customizable: Allows you to set the minimum percentage of observations before and after the breakpoint.
* Visualization: Plots the time series with the most significant breakpoint marked.
* Datetime Index: Maintains the datetime index from the input series for meaningful breakpoint identification.
