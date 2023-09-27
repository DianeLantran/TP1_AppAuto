import scikit
import dataTreatmentUtils
import mathsUtils
import time
import comparaison
import kbest
import dataViz
import preprocessing as prep
import pandas as pd

# Study of the dataSet
# dataViz.studyDataSet()

# Load file
FILE_PATH = "AirplaneCrashes.csv"
DATASET = pd.read_csv(FILE_PATH)

df = prep.preprocess(DATASET)

# PCA Custom

start_time = time.time()
# les valeurs propres < 5% ne sont pas prises en compte
df_customPCA = mathsUtils.PCA(df, 0.9)
end_time = time.time()
elapsed_time_custom = end_time - start_time


# Visualisation du nouveau dataset
# dataViz.plotDataSet(df_customPCA)


# Scikit-learn PCA
start_time = time.time()
dfScikit = scikit.applyPCA(df)
end_time = time.time()
elapsed_time2 = end_time - start_time

# Print informations
print(f"Temps pour la PCA custom : {elapsed_time_custom} secondes")
print(f"Temps pour la PCA de scikit : {elapsed_time2} secondes")


comparaison.colonnes(df)
comparaison.colonnes(dfScikit)
comparaison.colonnes(df_customPCA)
# comparaison.correlation(df, "bfore PCA")
# comparaison.correlation(dfScikit, "with Scikit")
# comparaison.correlation(df_customPCA, "with custom function")

# Kbest
# start_time = time.time()
# df_kbest = kbest.crop(df)
# end_time = time.time()
# elapsed_time3 = end_time - start_time
# print(f"Durée écoulée : {elapsed_time3} secondes")
# comparaison.correlation(df_kbest, "with k-best using chi2")
