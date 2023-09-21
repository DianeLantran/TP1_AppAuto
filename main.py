import readingFileUtils
import dataTreatmentUtils
import pandas as pd
from tabulate import tabulate
import preprocessing as prep

FILE_PATH = "data/AirplaneCrashes.csv"

df = pd.read_csv (FILE_PATH)

# Preprocessing
df = prep.simplifyDate(df)
df = prep.simplifyLocation(df)
df = prep.simplifyTime(df)
df = prep.colToOrdinal(df, ["Location", "Operator"])
# main

i = 2
j = 3
try:
    readingFileUtils.getVariable(i, j, FILE_PATH)
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
