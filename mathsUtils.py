"""
Created on Thu Sep 21 2023

@author: diane
"""

import pandas as pd
import numpy as np
from scipy.linalg import eigh


def covarianceMat(dataset):
    covM = dataset.cov()
    len(covM)
    return covM

def dim_red(threshold, eigenvalues, eigenvectors):
    tot = sum(eigenvalues)
    valid_eigenvectors = []
    valid_eigenvalues = []
    for i in range(len(eigenvalues)):
        if eigenvalues[i]/tot > threshold :
            valid_eigenvectors.append(eigenvectors[:,i].tolist())
            valid_eigenvalues.append(eigenvalues[i])
    return valid_eigenvectors, valid_eigenvalues

def vect_P(matrix):
    eigenvalues, eigenvectors = eigh(matrix)
    return(eigenvalues, eigenvectors)

def sort_vectP(eigenvalues, eigenvectors):
    #trie les vecteurs propres par ordre décroissants des valeur propres correspondantes
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort A and get indices in descending order
    sorted_eigenvectors = np.array(eigenvectors)[sorted_indices]
    return(sorted_eigenvectors)

def remove_zero_columns(matrix):
    # trouve les colones remplies par des 0
    non_zero_columns = np.any(matrix != 0, axis=0)
    # retourne la matrice sans les colones remplies de 0
    result = matrix[:, non_zero_columns]
    
    return result

def PCA(dataset, threshold):
    cov = covarianceMat(dataset)
    eigenvalues, eigenvectors = vect_P(cov)
    featureVect, new_eigen_values = dim_red(threshold, eigenvalues, eigenvectors)
    newData = np.dot(featureVect, dataset.T)
    return(newData)


