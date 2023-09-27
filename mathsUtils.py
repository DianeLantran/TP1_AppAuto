"""
Created on Thu Sep 21 2023

@author: diane
"""

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def covarianceMat(dataset):
    covM = dataset.cov()
    return covM

def dim_red(threshold, eigenvalues, eigenvectors):
    sum_eigenvalues = sum(eigenvalues)
    valid_eigenvectors = []
    valid_eigenvalues = []
    total = 0
    i = 0
    while total < threshold:
        valid_eigenvectors.append(eigenvectors[i].tolist())
        valid_eigenvalues.append(eigenvalues[i])
        total += eigenvalues[i]/sum_eigenvalues
        i += 1
    return valid_eigenvectors, valid_eigenvalues

def vect_P(matrix):
    eigenvalues, eigenvectors = eigh(matrix)
    return(eigenvalues, eigenvectors)

def sort_vectP(eigenvalues, eigenvectors):
    #trie les vecteurs propres par ordre décroissants des valeur propres correspondantes
    sorted_indices, sorted_eigenvalues = zip(*sorted(enumerate(eigenvalues),
                                                     key=lambda x: x[1],
                                                     reverse=True))
    sorted_eigenvectors = eigenvectors[np.array(sorted_indices)]
    return(sorted_eigenvalues, sorted_eigenvectors)

def PCA(dataset, threshold):
    cov = covarianceMat(dataset)
    eigenvalues, eigenvectors = vect_P(cov)
    
    # Affichage du diagramme de la variance expliquée par colonne avant réduction
    eigenvalue_proportion = eigenvalues / np.sum(eigenvalues)
    plt.bar(range(len(eigenvalues)), eigenvalue_proportion, tick_label=[f'Colonne {i+1}' for i in range(len(eigenvalues))])
    plt.title('Explained variances per column')
    plt.xlabel('Columns')
    plt.ylabel('Percentage of explained variances')
    plt.xticks(rotation=80)
    plt.show()
    
    sorted_eigenvalues, sorted_eigenvectors = sort_vectP(eigenvalues, eigenvectors)
    featureVect, new_eigen_values = dim_red(threshold, sorted_eigenvalues, sorted_eigenvectors)
    
    # Affichage du diagramme de la variance expliquée par colonne après réduction
    eigenvalue_proportion = new_eigen_values / np.sum(new_eigen_values)
    plt.bar(range(len(new_eigen_values)), eigenvalue_proportion, tick_label=[f'{i+1}' for i in range(len(new_eigen_values))])
    plt.title('Explained variances per column')
    plt.xlabel('Columns')
    plt.ylabel('Percentage of explained variances')
    plt.show()
    
    # Affichage du diagramme de la variance expliquée cumulée
    cumulative_eigenvalue_proportion = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)
    plt.bar(range(1, len(sorted_eigenvalues) + 1), cumulative_eigenvalue_proportion, tick_label=[f'{i+1}' for i in range(len(sorted_eigenvalues))])
    plt.title('Cumulative explained variances')
    plt.xlabel('Columns')
    plt.ylabel('Cumulative percentage of explained variances')
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% Explained Variance')
    plt.show()
    
    newData = np.dot(dataset, np.array(featureVect).T)
    return(newData)