import csv
import readingFileUtils
import dataTreatmentUtils
import mathsUtils
import pandas as pd
from tabulate import tabulate
import preprocessing as prep

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif  # Vous pouvez utiliser une autre fonction de score

FILE_PATH = "AirplaneCrashes.csv"
DATASET = pd.read_csv(FILE_PATH)

# nettoyage des données (<70% de données sur lignes et colones)
df = dataTreatmentUtils.removeUselessColumns(DATASET, 30)
df = df.drop("Summary", axis=1)
df = df.drop("Registration", axis=1)
df = dataTreatmentUtils.removeUselessRows(df, 25)

# Preprocessing
df = prep.simplifyDate(df)
df = prep.simplifyLocation(df)
df = prep.simplifyRoute(df)
df = prep.colToOrdinal(df, ["Location", "Operator", 
                            "AC Type", "Departure", 
                            "Arrival", "cn/ln"])

# Standardization
df = prep.standardize(df)

# PCA
reducData_PCA = mathsUtils.PCA(df, 0.05) #les valeurs propres < 5% ne sont pas prises en compte
print(reducData_PCA)

#Comparaison avec SelectKBest
k_best = SelectKBest(score_func=f_classif, k=len(df.columns))  # k = nombre de caractéristiques souhaité
reducData_SB = k_best.fit_transform(df, None) #étiquettes de classe cibles = None -> non supervisé
print(reducData_SB)