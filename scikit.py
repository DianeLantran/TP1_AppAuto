import preprocessing as prep
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


def removeMostEmptyData(dataSet):
    # Remove columns with less than 70% of data
    threshold = 0.7
    dfScikit = dataSet.dropna(thresh=int(threshold * dataSet.shape[0]), axis=1)
    # Drop specific columns
    dfScikit = dfScikit.drop(["Summary", "Registration"], axis=1)
    # Remove rows with less than 30% of data
    dfScikit = dfScikit.dropna(thresh=int(0.30 * dfScikit.shape[1]))
    return dfScikit


def preprocessing(df):
    df = prep.simplifyDate(df)
    df = prep.simplifyLocation(df)
    df = prep.simplifyRoute(df)
    # Scikit Encoder
    colnames = ["Location", "Operator",
                "AC Type", "Departure",
                "Arrival", "cn/ln"]
    encoder = OrdinalEncoder()
    df[colnames] = encoder.fit_transform(df[colnames].values.reshape(-1, 1))

    return df


def standardization(df):
    scaler = StandardScaler()
    df_standardized = scaler.fit_transform(df)
    return df_standardized


def fillNA(df):
    # Create an imputer
    imputer = SimpleImputer(strategy='mean')
    # Fit and transform the imputer on your data
    df_no_nan = imputer.fit_transform(df)
    return df_no_nan


def applyPCA(df):
    # PCA with explained variance threshold (5% in your case)
    explained_variance_threshold = 0.05
    pca = PCA(explained_variance_threshold)
    # Fit PCA and transform the data (corrected)
    reducData_PCA = pca.fit_transform(df)
    return reducData_PCA