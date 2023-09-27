import scikit
import dataTreatmentUtils
import mathsUtils
import pandas as pd
import preprocessing as prep
import time
import comparaison
import kbest
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
# Vous pouvez utiliser une autre fonction de score
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest


# Load file
FILE_PATH = "AirplaneCrashes.csv"
DATASET = pd.read_csv(FILE_PATH)

# Columns renaming
column_name_mapping = {
    'Aboard': 'Total aboard',
    'Aboard Passangers': 'Passengers aboard',
    'Aboard Crew': 'Crew aboard',
    'Fatalities': 'Total fatalities',
    'Fatalities Passangers': 'Passengers fatalities',
    'Fatalities Crew': 'Crew fatalities'
}

# Rename the columns using the mapping
df = DATASET.rename(columns=column_name_mapping)

# Missing data deletion
# Removing columns with more than 30% NA values
df = dataTreatmentUtils.removeUselessColumns(df, 30)
# Removing unprocessable columns
df = df.drop(["Summary", "Registration"], axis=1)
# Removing rows with more than 25% NA values
df = dataTreatmentUtils.removeUselessRows(df, 25)

# Discretization
df = prep.discretization(df)

# Missing data replacement
df = prep.replaceMissingCrewPassengers(df)

# Ordinal columns encoding
df = prep.encodeOrdinalColumns(df)

# Standardization
df = prep.standardize(df)

# PCA
start_time = time.time()
# les valeurs propres < 5% ne sont pas prises en compte
reducData_PCA = mathsUtils.PCA(df, 0.9)
end_time = time.time()
elapsed_time1 = end_time - start_time
len1 = len(reducData_PCA)
print(f"Size of our dataSet when using our custom-made functions: {len1}")
# Affichez la durée écoulée
print(f"Durée écoulée : {elapsed_time1} secondes")

# #Comparaison avec SelectKBest
# k_best = SelectKBest(score_func=f_classif, k=len(df.columns))  # k = nombre de caractéristiques souhaité
# reducData_SB = k_best.fit_transform(df, None) #étiquettes de classe cibles = None -> non supervisé
# print(reducData_SB)

# Visualisation du nouveau dataset
reduced_df = pd.DataFrame(reducData_PCA)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(reduced_df[reduced_df.columns[0]], reduced_df[reduced_df.columns[1]])
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('Visualization of data')
plt.show()

# Comparaison avec scikit-learn

dfScikit = scikit.removeMostEmptyData(DATASET)

dfScikit = scikit.preprocessing(dfScikit)

dfScikit = scikit.standardization(dfScikit)

dfScikit = scikit.fillNA(dfScikit)

start_time = time.time()
dfScikit = scikit.applyPCA(df)
end_time = time.time()
elapsed_time2 = end_time - start_time

# Print informations
num_samples = dfScikit.shape[0]
print(
    f"Size of our dataSet when using scikit functions: {num_samples}")
# Affichez la durée écoulée
print(f"Durée écoulée : {elapsed_time2} secondes")

comparaison.colonnes(df)
comparaison.colonnes(dfScikit)
comparaison.colonnes(reducData_PCA)
# comparaison.correlation(df, "bfore PCA")
# comparaison.correlation(dfScikit, "with Scikit")
# comparaison.correlation(reducData_PCA, "with custom function")

# Kbest
start_time = time.time()
df_kbest = kbest.crop(df)
end_time = time.time()
elapsed_time3 = end_time - start_time
print(f"Durée écoulée : {elapsed_time3} secondes")
print(df_kbest.dtypes)
comparaison.correlation(df_kbest, "with k-best using chi2")
