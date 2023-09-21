# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:26:08 2023

@author: basil
"""
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def splitDate(date):
    month, day, year = map(int, date.split('/'))
    return month, year

def simplifyDate(df):
    df[['Month', 'Year']] = df['Date'].apply(lambda x: pd.Series(splitDate(x)))
    df = df.drop('Date', axis=1)
    return df

def getBroadestLocation(location):
    return str(location).split(",")[-1].strip()

def simplifyLocation(df):
    # Get broadest location
    df['Location'] = df['Location'].apply(getBroadestLocation)  
    return df

def splitTime(time):
    if pd.notna(time):
        hour, minute = map(int, time.split(':'))
    else:
        hour = pd.NA
    return hour

def simplifyTime(df):
    df['Hour'] = df['Time'].apply(splitTime)
    df = df.drop('Time', axis=1)
    return df

def colToOrdinal(df, colnames):
    # Prepare encoder object
    encoder = OrdinalEncoder()
    
    # Fit and transform the selected column
    df[colnames] = encoder.fit_transform(df[colnames])
    return df