import csv


def covarianceMat(dataset):
    covM = dataset.cov()
    return covM