import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


def crop(df):
    k = 8  # Number of columns to keep
    chi2_scores = []
    for col1 in df.columns:
        chi2_row = []
        for col2 in df.columns:
            if col1 == col2:
                # Ignorer la corrélation avec la même caractéristique
                chi2_row.append(0.0)
            else:
                contingency_table = pd.crosstab(df[col1], df[col2])
                chi2, _, _, _ = chi2_contingency(contingency_table)
                chi2_row.append(chi2)
        chi2_scores.append(chi2_row)

    # Convertir la liste en une matrice numpy
    chi2_matrix = np.array(chi2_scores)

    # Sélectionner les 'k' caractéristiques avec les scores du Chi-square les plus élevés
    selected_features_indices = np.unravel_index(
        np.argsort(chi2_matrix, axis=None), chi2_matrix.shape)
    selected_feature_indices = selected_features_indices[1][:k]

    # Sélectionner les colonnes correspondant aux caractéristiques sélectionnées
    selected_features = df.columns[selected_feature_indices]

    # Créer un nouveau DataFrame avec les caractéristiques sélectionnées
    df_selected = df[selected_features]

    return df_selected
