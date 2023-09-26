# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:26:08 2023

@author: basil
"""
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

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

def encodeOrdinalColumns(df, colnames):
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

def replaceMissingCrewPassengers(df):
    cols_aboard = ["Total aboard", "Passengers aboard", "Crew aboard"]
    cols_fatalities =["Total fatalities", "Passengers fatalities", 
                     "Crew fatalities"]
    aboard_medians = df[cols_aboard].median()
    fatalities_medians = df[cols_fatalities].median()
    
    for index, row in df.iterrows():
        aboard_null_values_count = len(cols_aboard) - row[cols_aboard].count()
        fatalities_null_values_count = len(cols_fatalities) - row[cols_fatalities].count()
        match aboard_null_values_count:
            case 2:
                pos = np.where(row[cols_aboard].isnull())[0][0]
                df.loc[index, cols_aboard[pos]] = aboard_medians[pos]
            case 3:
                df.loc[index, cols_aboard[0]] = aboard_medians[0]
                df.loc[index, cols_aboard[2]] = aboard_medians[2]
        match fatalities_null_values_count:
            case 2:
                pos = np.where(row[cols_fatalities].isnull())[0][0]
                df.loc[index, cols_fatalities[pos]] = fatalities_medians[pos]
            case 3:
                df.loc[index, cols_fatalities[0]] = fatalities_medians[0]
                df.loc[index, cols_fatalities[2]] = fatalities_medians[2]
    df['Crew fatalities'] = df['Crew fatalities'].fillna(df['Total fatalities'] - df["Passengers fatalities"])                
    df['Crew aboard'] = df['Crew aboard'].fillna(df['Total aboard'] - df["Passengers aboard"])                
    df['Passengers aboard'] = df['Passengers aboard'].fillna(df['Total aboard'] - df["Crew aboard"])
    df['Passengers fatalities'] = df['Passengers fatalities'].fillna(df['Total fatalities'] - df["Crew fatalities"])
    return df

