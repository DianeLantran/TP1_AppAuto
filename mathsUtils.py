"""
Created on Thu Sep 21 2023

@author: diane
"""

import pandas as pd
from scipy.linalg import eigh


def covarianceMat(dataset):
    covM = dataset.cov()
    return covM

def val_vect_Propres(matrix):
    eigenvalues, eigenvectors = eigh(matrix)
