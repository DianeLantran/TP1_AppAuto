import csv
import readingFileUtils
import dataTreatmentUtils
import pandas as pd
from tabulate import tabulate
file_path = "AirplaneCrashes.csv"

# main

i = 2
j = 3
try:
    readingFileUtils.getVariable(i, j, file_path)
except FileNotFoundError:
    print("File not found or cannot be created.")
except PermissionError:
    print("Permission denied to create or write to the file.")
except Exception as e:
    print(f"An error occurred: {e}")
    
    

    
# nettoyage des données
df = pd.read_csv(file_path)
#print(df)
df = dataTreatmentUtils.removeUselessColumns(df, 30)
df = df.drop("Summary", axis=1)
#print(df)
df = dataTreatmentUtils.removeUselessRows(df, 25)
#print(df)
