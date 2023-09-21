import csv
import readingFileUtils
import dataTreatmentUtils
import mathsUtils
import pandas as pd
from tabulate import tabulate
import preprocessing as prep

FILE_PATH = "AirplaneCrashes.csv"
DATASET = pd.read_csv(FILE_PATH)

#fonction pour lire les données
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
    
    

    
# nettoyage des données (<70% de données sur lignes et colones)
df = dataTreatmentUtils.removeUselessColumns(DATASET, 30)
df = df.drop("Summary", axis=1)
df = dataTreatmentUtils.removeUselessRows(df, 25)

# Preprocessing
df = prep.simplifyDate(df)
df = prep.simplifyLocation(df)
df = prep.colToOrdinal(df, ["Location", "Operator", "AC Type"])

# PCA
cov = mathsUtils.covarianceMat(df) #decommenter apres la discretisation