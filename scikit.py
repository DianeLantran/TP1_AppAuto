from sklearn.decomposition import PCA


def applyPCA(df):
    """
    Applying Principal Component Analysis using scikit-learn method 

    Arguments : 
    df: pandas dataframe

    Returns: 
    pca_data: pandas dataframe
    """
    explained_variance_threshold = 0.9
    pca = PCA(explained_variance_threshold)
    # Fit PCA and transform the data (corrected)
    reducData_PCA = pca.fit_transform(df)
    return reducData_PCA
