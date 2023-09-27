import tkinter as tk
from tkinter import scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def colonnes(df):
    nombre_de_colonnes = df.shape[1]
    print("Nombre de colonnes :", nombre_de_colonnes)

    return


def correlation(df, titre):
    correlation_matrix = np.corrcoef(df, rowvar=False)

    # Affichage matrice
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.matshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    colorbar = fig.colorbar(heatmap)
    ax.set_xticks(np.arange(len(correlation_matrix)))
    ax.set_yticks(np.arange(len(correlation_matrix)))
    ax.set_xticklabels(range(1, len(correlation_matrix) + 1))
    ax.set_yticklabels(range(1, len(correlation_matrix) + 1))
    plt.title("Correlation matrix for each column "+titre)
    plt.show()
    return
