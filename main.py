import dataTreatmentUtils
import mathsUtils
import pandas as pd
import preprocessing as prep
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif  # Vous pouvez utiliser une autre fonction de score

# Load file
FILE_PATH = "AirplaneCrashes.csv"
df = pd.read_csv(FILE_PATH)

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
df = df.rename(columns=column_name_mapping)
save = df.copy()
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
cols_with_missing_values = missing_values_mask[missing_values_mask].index.tolist()
df = prep.encodeOrdinalColumns(df, cols_with_missing_values + ["Location"])

# Standardization
df = prep.standardize(df)

# PCA
reducData_PCA = mathsUtils.PCA(df, 0.05) #les valeurs propres < 5% ne sont pas prises en compte
print(reducData_PCA)

#Comparaison avec SelectKBest
k_best = SelectKBest(score_func=f_classif, k=len(df.columns))  # k = nombre de caractéristiques souhaité
reducData_SB = k_best.fit_transform(df, None) #étiquettes de classe cibles = None -> non supervisé
print(reducData_SB)