# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:26:08 2023

@author: basil
"""
import pandas as pd
import re
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


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

def splitRoute(route):
    if (pd.notna(route)):
        routeArray = re.split(r'(\s?-\s)|(\s-\s?)', route.strip())
        routeArray = [routeArray[0], routeArray[-1]]
        departure, arrival = list(map(lambda x: x.split(",")[0], routeArray))
        #print(re.split(r'\s-\s', route.strip()))
    else:
        departure = arrival = float("nan")
    return departure, arrival
        

def simplifyRoute(df):
    df[['Departure', 'Arrival']] = df['Route'].apply(lambda x: pd.Series(splitRoute(x)))
    df = df.drop('Route', axis=1)
    return df

def colToOrdinal(df, colnames):
    # Prepare encoder object
    encoder = OrdinalEncoder(encoded_missing_value=-1)
    
    # Fit and transform the selected column
    df[colnames] = encoder.fit_transform(df[colnames])
    return df

def standardize(df):
    scaler = StandardScaler()
    df_standardized = scaler.fit_transform(df)
    df_standardized = pd.DataFrame(df_standardized, columns=df.columns)
    return df_standardized