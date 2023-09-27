# -*- coding: utf-8 -*-
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


def splitDate(date):
    """
    Splits a date and returns the month and year.

    Parameters:
    date: The date in MM/DD/YYYY format to split.

    Returns:
    (int, int): The month and the year of the date.
    """
    month, day, year = map(int, date.split('/'))
    return month, year


def simplifyDate(df):
    """
    Removes the date column in the Dataframe and puts
    a month and year columns instead.

    Parameters:
    df: The Dataframe to process.

    Returns:
    Dataframe: The updated Dataframe.
    """
    df[['Month', 'Year']] = df['Date'].apply(lambda x: pd.Series(splitDate(x)))
    df = df.drop('Date', axis=1)
    return df


def getBroadestLocation(location):
    """
    Gets the most general location in a specific location by taking the 
    last element in the location.

    Parameters:
    location: The location to split (separated by commas).

    Returns:
    str: The last element of the location parameter.
    """
    return str(location).split(",")[-1].strip()


def simplifyLocation(df):
    """
    Replaces the values in the Location column by the last element in the 
    original value

    Parameters:
    df: The Dataframe to process.

    Returns:
    Dataframe: The updated Dataframe.
    """
    # Get broadest location
    df['Location'] = df['Location'].apply(getBroadestLocation)
    return df


def splitRoute(route):
    """
    Split a route to retreive the start and end of the route. If the 
    parameter isn't a route in the wanted format, the element will be 
    duplicated in the returned results

    Parameters:
    route: a string preferably in the format "departure - ... - arrival"

    Returns:
    (str, str): The departure and arrival of the route
    """
    if (pd.notna(route)):
        # Split the route and get the first and last element, then remove
        # useless information
        routeArray = re.split(r'(\s?-\s)|(\s-\s?)', route.strip())
        routeArray = [routeArray[0], routeArray[-1]]
        departure, arrival = list(map(lambda x: x.split(",")[0], routeArray))
    else:
        # If no value is found, we return two nan values
        departure = arrival = float("nan")
    return departure, arrival


def simplifyRoute(df):
    """
    Removes the Route column in the Dataframe and puts
    the start and end of the route in eponym columns instead.

    Parameters:
    df: The Dataframe to process.

    Returns:
    Dataframe: The updated Dataframe.
    """
    df[['Departure', 'Arrival']] = df['Route'].apply(
        lambda x: pd.Series(splitRoute(x)))
    df = df.drop('Route', axis=1)
    return df


def encodeOrdinalColumns(df):
    """
    Encodes the given columns to remove the ordinal values and have numbers 
    instead. NA values will be encoded with -1

    Parameters:
    df: The Dataframe to process.
    colnames: An array containing the names of the columns to encode

    Returns:
    Dataframe: The updated Dataframe.
    """

    missing_values_mask = df.isnull().any()
    cols_with_missing_values = missing_values_mask[missing_values_mask].index.tolist(
    )

    # Prepare encoder object
    encoder = OrdinalEncoder(encoded_missing_value=-1)

    # Fit and transform the selected column
    colnames = cols_with_missing_values + ["Location"]
    df[colnames] = encoder.fit_transform(df[colnames])
    return df


def standardize(df):
    """
    Standardizes the dataframe with a standard standardization

    Parameters:
    df: The Dataframe to standardize

    Returns:
    Dataframe: The updated Dataframe.
    """
    column_means = df.mean()
    column_std = df.std()
    standardized_df = (df - column_means) / column_std
    return standardized_df


def replaceMissingCrewPassengers(df):
    """
    Replaces the NA values in the columns related to passengers and crew in the
    dataframe

    Parameters:
    df: The Dataframe to process

    Returns:
    Dataframe: The updated Dataframe.
    """
    # Group of columns to process together
    cols_aboard = ["Total aboard", "Passengers aboard", "Crew aboard"]
    cols_fatalities = ["Total fatalities", "Passengers fatalities",
                       "Crew fatalities"]
    # Medians computing, this is preferable to mean because of some outliers
    aboard_medians = df[cols_aboard].median()
    fatalities_medians = df[cols_fatalities].median()

    for index, row in df.iterrows():
        # For each row, we compute which columns (amongst the aboard columns),
        # have NA values and we replace one or 2 of them by the median of the
        # column
        aboard_null_values_count = len(cols_aboard) - row[cols_aboard].count()
        match aboard_null_values_count:
            case 2:
                pos = np.where(row[cols_aboard].isnull())[0][0]
                df.loc[index, cols_aboard[pos]] = aboard_medians[pos]
            case 3:
                df.loc[index, cols_aboard[0]] = aboard_medians[0]
                df.loc[index, cols_aboard[2]] = aboard_medians[2]

        # Similar work is done for the fatalities
        fatalities_null_values_count = len(
            cols_fatalities) - row[cols_fatalities].count()
        match fatalities_null_values_count:
            case 2:
                pos = np.where(row[cols_fatalities].isnull())[0][0]
                df.loc[index, cols_fatalities[pos]] = fatalities_medians[pos]
            case 3:
                df.loc[index, cols_fatalities[0]] = fatalities_medians[0]
                df.loc[index, cols_fatalities[2]] = fatalities_medians[2]

    # Now only one column out of the three should be empty, so we compute its
    # value with the other columns
    df['Crew fatalities'] = df['Crew fatalities'].fillna(
        df['Total fatalities'] - df["Passengers fatalities"])
    df['Crew aboard'] = df['Crew aboard'].fillna(
        df['Total aboard'] - df["Passengers aboard"])
    df['Passengers aboard'] = df['Passengers aboard'].fillna(
        df['Total aboard'] - df["Crew aboard"])
    df['Passengers fatalities'] = df['Passengers fatalities'].fillna(
        df['Total fatalities'] - df["Crew fatalities"])
    return df


def discretization(df):
    """
    Do all the steps of the discretization 

    Parameters:
    df: The Dataframe to process

    Returns:
    Dataframe: The updated Dataframe.
    """
    df = simplifyDate(df)
    df = simplifyLocation(df)
    df = simplifyRoute(df)
    return (df)
