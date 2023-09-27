"""
Created on Thu Sep 21 2023

@author: diane
"""

import numpy as np
from scipy.linalg import eigh

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
    sorted_eigenvalues, sorted_eigenvectors = sort_vectP(eigenvalues, eigenvectors)
    featureVect, new_eigen_values = dim_red(threshold, sorted_eigenvalues, sorted_eigenvectors)
    newData = np.dot(dataset, np.array(featureVect).T)
    return(newData)