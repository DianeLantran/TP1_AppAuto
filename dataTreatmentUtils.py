import pandas as pd

def getMissingDataPercentageByColumn(dataset):
    # Calculer le nombre de données manquantes par colonne
    missing_data = dataset.isnull().sum()

    # Calculer le pourcentage de données manquantes par colonne
    percentage = (missing_data / len(dataset)) * 100

    # Créer un DataFrame pour afficher les résultats
    missing_data_tab = pd.DataFrame({
        'Colonne': dataset.columns,
        'Données manquantes': missing_data,
        'Pourcentage de données manquantes': percentage
    })
    return missing_data_tab
    
    
def deleteUselessColumns(dataset, missing_data_tab):
    #Suppression des colonnes avec trop de valeurs nulles
    for i in range(len(missing_data_tab)):
        if missing_data_tab.iloc[i, 2] >= 30  or missing_data_tab.iloc[i, 0] == "Summary":
            dataset = dataset.drop(missing_data_tab.iloc[i, 0], axis=1)
    return dataset
            