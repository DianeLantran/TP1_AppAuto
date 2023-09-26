# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import preprocessing as prep
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def plotHist(data, title, xlabel, ylabel, bins, cmap = "Purples"):
    # Get hist values
    hist, bins, _ = plt.hist(data, bins=bins, alpha = 0)

    # Get color map
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

def plotDateHist(df):
    df[['Month', 'Year']] = df['Date'].apply(lambda x: pd.Series(prep.splitDate(x)))
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Generate and save the first plot
    plt.sca(axs[0])
    plotHist(df["Year"], "Evolution of airplane crashes between 1908-2019", 
             "Year", "Number of crashes", 8)
    
    # Generate and save the second plot
    plt.sca(axs[1])
    plotHist(df["Month"], "Repartition of airplane crashes by month", 
             "Year", "Number of crashes", 12)
    
    # Adjust layout and display the subplots
    plt.tight_layout()
    plt.show()
    

def splitTime(time):
    if pd.notna(time):
        hour, minute = map(int, time.split(':'))
    else:
        hour = pd.NA
    return hour
    
def plotNA(df):
    na_counts_array = df.isna().sum().values
    not_na_counts_array = df.count().values
    colnames = df.columns.tolist()
    colors = ["#D4403D", "#13B035"]
    stack_data = {
            "NA values": na_counts_array,
            "Non NA values": not_na_counts_array
    }
    width = 0.5

    fig, ax = plt.subplots(figsize = (35, 6))
    bottom = np.zeros(len(colnames))

    for i, (boolean, weight_count) in enumerate(stack_data.items()):        
        ax.bar(colnames, weight_count, 
               width, label=boolean, bottom=bottom, color = colors[i])
        bottom += weight_count
    
    target_value = df.Time.count() * 30 / 100
    ax.axhline(y=target_value, color='gray', 
               linestyle='--', label=f'Threshold at 30% ({target_value})')

    ax.set_title("Repartition of the NA/non NA values for each variable")
    ax.legend(loc="upper right")
    plt.show()
    

def plotTimeHist(df):
    df[['Hour']] = df['Time'].apply(lambda x: pd.Series(splitTime(x)))
    
    condition = df['Hour'] < 24
    filtered_column = df.loc[condition, 'Hour']
    plotHist(filtered_column, "Repartition of airplane crashes by hour", 
             "Hour of the day", "Number of crashes", 24)    
    
def plotPie(data, title, threshold = 0):
    # Calculate the total number of values in the 'category' column
    total_values = len(data)
    
    # Set the threshold as a percentage of the total number of values
    # Adjust the threshold_percentage as needed (e.g., 10%)
    threshold_percentage = threshold
    threshold = total_values * (threshold_percentage / 100)
    
    # Count the occurrences of each value in the 'category' column
    value_counts = data.value_counts()
    
    # Group values that occur less than the threshold into an 'Others' category
    value_counts['Others'] = value_counts[value_counts < threshold].sum()
    value_counts = value_counts[value_counts >= threshold]
    
    # Plot a pie chart
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%',
            textprops={'fontsize': 14})
    plt.title(title)
    
def plotUniqueValues(df):

    # Calculate the number of unique values for each column
    unique_value_counts = df.nunique()
    
    # Create a gradient of colors from green to red based on the number of unique values
    colors = plt.cm.get_cmap('RdYlGn_r')(unique_value_counts / max(unique_value_counts))
    #custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=len(unique_value_counts))
    
    # Generate the bars and colors
    plt.figure(figsize=(13, 5))
    bars = plt.bar(unique_value_counts.index, unique_value_counts, color=colors)
    
    # Add the count text on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom')
    
    plt.xlabel('Variables')
    plt.ylabel('Number of Unique Values')
    plt.title('Number of Unique Values for Each Variable')
    plt.xticks(rotation=80)
    plt.show()

    
FILE_PATH = "AirplaneCrashes.csv"
df = pd.read_csv(FILE_PATH)
plotUniqueValues(df)
df = prep.simplifyLocation(df)
plotDateHist(df)
plotTimeHist(df)
plotNA(df)
fig, axs = plt.subplots(1, 1, figsize=(20, 20))
plotPie(df["Location"], "Repartition of the crash location", 1.5)
plt.show()
fig, axs = plt.subplots(1, 1, figsize=(20, 20))
plotPie(df["Operator"], "Repartition of the airplane operators", 0.9)
plt.show()
df = prep.simplifyRoute(df)
fig, axs = plt.subplots(1, 2, figsize=(20, 20))
plt.sca(axs[0])
plotPie(df["Departure"], "Repartition of the flight departures", 0.7)
plt.sca(axs[1])
plotPie(df["Arrival"], "Repartition of the flight arrivals", 0.7)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(20, 20))
plotPie(df["AC Type"], "Repartition of the aircraft types", 0.7)
plt.show()

plotUniqueValues(df)