"""
Created on Thu Sep 21 2023

@author: diane
"""

import pandas as pd
import numpy as np
from scipy.linalg import eigh


def covarianceMat(dataset):
    covM = dataset.cov()
    return covM

def dim_red(threshold, eigenvalues, eigenvectors):
    tot = sum(eigenvalues)
    for i in range(len(eigenvalues)):
        if eigenvalues[i]/tot < threshold :
            eigenvalues[i] = 0
            eigenvectors[:,i] = np.zeros((len(eigenvectors[:,i]),))
    return 

def vect_P(matrix):
    eigenvalues, eigenvectors = eigh(matrix)
    return(eigenvalues, eigenvectors)

def sort_vectP(eigenvalues, eigenvectors):
    #trie les vecteurs propres par ordre décroissants des valeur propres correspondantes
    eigen_pairs = [(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))]
    eigen_pairs.sort(key=lambda x: x[0], reverse=True)
    sorted_eigenvectors = np.array([pair[1] for pair in eigen_pairs]).T
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
    dim_red(threshold, eigenvalues, eigenvectors) #met à 0 les colonnes trop peu significatives
    sorted_eigenvectors = sort_vectP(eigenvalues, eigenvectors)
    featureVect = remove_zero_columns(sorted_eigenvectors)
    newData = np.dot(dataset, featureVect)
    return(newData)


