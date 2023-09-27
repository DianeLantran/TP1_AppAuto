import scikit
import dataTreatmentUtils
import mathsUtils
import pandas as pd
import preprocessing as prep
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
# Vous pouvez utiliser une autre fonction de score
from sklearn.feature_selection import f_classif
import time


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
df = prep.simplifyDate(df)
df = prep.simplifyLocation(df)
df = prep.simplifyRoute(df)

# Missing data replacement
df = prep.replaceMissingCrewPassengers(df)

# Ordinal columns encoding
missing_values_mask = df.isnull().any()
cols_with_missing_values = missing_values_mask[missing_values_mask].index.tolist(
)
df = prep.encodeOrdinalColumns(df, cols_with_missing_values + ["Location"])

# Standardization
df = prep.standardize(df)

# PCA
start_time = time.time()
# les valeurs propres < 5% ne sont pas prises en compte
reducData_PCA = mathsUtils.PCA(df, 0.05)
print(reducData_PCA.shape)
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
reduced_df = pd.DataFrame(reducData_PCA, columns=['Colonne1', 'Colonne2', 'Colonne3', 'Colonne4', 'Colonne5', 'Colonne6', 'Colonne7', 'Colonne8', 'Colonne9'])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(reduced_df['Colonne1'], reduced_df['Colonne2'])
ax.set_xlabel('Composante principale 1')
ax.set_ylabel('Composante principale 2')
plt.show()

variance_origin = df.describe().iloc[2,:]
variance_pca = reduced_df.describe().iloc[2,:]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Première figure (stem 1)
ax1.stem(variance_origin)
ax1.set_title('Variance des colonnes du dataset avant PCA')
ax1.set_xlabel('Colonnes')
ax1.set_ylabel('Variance')

# Deuxième figure (stem 2)
ax2.stem(variance_pca)
ax2.set_title('Variance des colonnes du dataset après PCA')
ax2.set_xlabel('Colonnes')
ax2.set_ylabel('Variance')

# Comparaison avec scikit-learn

dfScikit = scikit.removeMostEmptyData(DATASET)

dfScikit = scikit.preprocessing(dfScikit)

dfScikit = scikit.standardization(dfScikit)

dfScikit = scikit.fillNA(dfScikit)

start_time = time.time()
dfScikit = scikit.applyPCA(dfScikit)
end_time = time.time()
elapsed_time2 = end_time - start_time

# Print informations
num_samples = dfScikit.shape[0]
print(
    f"Size of our dataSet when using scikit functions: {num_samples}")
# Affichez la durée écoulée
print(f"Durée écoulée : {elapsed_time2} secondes")
