import pandas as pd


#Traitement des colonnes

def getMissingDataPercentagePerColumn(dataset):
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
    
def removeUselessColumns(dataset, max_percentage):
    missing_data_tab = getMissingDataPercentagePerColumn(dataset)
    #Suppression des colonnes avec trop de valeurs nulles
    for i in range(len(missing_data_tab)):
        if missing_data_tab.iloc[i, 2] >= 30:
            dataset = dataset.drop(missing_data_tab.iloc[i, 0], axis=1)
    return dataset


#Traitement des lignes

def getMissingDataPercentageForOneRow(row):
    missing_data = row.isnull().sum()
    percentage = (missing_data / len(row)) * 100
    return percentage

def removeUselessRows(dataset, max_percentage):
    dataset = dataset[dataset.apply(getMissingDataPercentageForOneRow, axis=1) <= max_percentage]
    return dataset   

