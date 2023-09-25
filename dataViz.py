# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 01:04:09 2023

@author: basil
"""

import pandas as pd
import matplotlib.pyplot as plt
import preprocessing as prep
import numpy as np

def plotHist(data, title, xlabel, ylabel, bins, cmap = "Purples"):
    hist, bins, _ = plt.hist(data, bins=bins, alpha=0)

    # Get color map for purples
    cmap = plt.get_cmap(cmap)
    
    # Normalize the frequencies to map to the colormap
    hist_normalized = hist / hist.max()
    
    # Create a color map based on normalized frequencies
    colors = cmap(hist_normalized)
    
    # Create a histogram with custom colors
    plt.bar(bins[:-1], hist, width=np.diff(bins), 
            color=colors, alpha=0.7, edgecolor = "black")
    
    
    # Labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Overall #
    plt.title(title)
    
    plt.show()

def plotDateHist(df):
    df[['Month', 'Year']] = df['Date'].apply(lambda x: pd.Series(prep.splitDate(x)))
    
    plotHist(df["Year"], "Evolution of airplane crashes between 1908-2019", 
             "Year", "Number of crashes", 8)
    plotHist(df["Month"], "Repartition of airplane crashes by month", 
             "Year", "Number of crashes", 12)
    

def splitTime(time):
    if pd.notna(time):
        hour, minute = map(int, time.split(':'))
    else:
        hour = pd.NA
    return hour
    
def plotNAValues(variable, variableName: str):
    nbNA = variable.isna().sum()
    nbNotNA = len(variable) - nbNA
    plt.pie([nbNotNA, nbNA], labels= ["Non NA values", "NA values"], 
            colors = ["green", "red"], autopct='%1.1f%%', shadow=True, 
            startangle = 200)
    
    plt.title("Repartition of NA/Non-NA values in the " + str(variableName )+ " variable")
    plt.show()
    

def plotTimeHist(df):
    df[['Hour']] = df['Time'].apply(lambda x: pd.Series(splitTime(x)))
    
    condition = df['Hour'] < 24
    filtered_column = df.loc[condition, 'Hour']
    plotHist(filtered_column, "Repartition of airplane crashes by hour", 
             "Hour of the day", "Number of crashes", 24)
    
    plotNAValues(df["Time"], "Time")
    
    

FILE_PATH = "AirplaneCrashes.csv"
df = pd.read_csv(FILE_PATH)
plotDateHist(df)
plotTimeHist(df)