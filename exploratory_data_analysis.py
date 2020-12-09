import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_descriptive_statistics(df):
    print("Dataset Size:\n")
    print(f"There are {df.shape[0]} rows and {df.shape[1]} columns in the dataset.")
    print()
    print("---------------------------------------------------")
    print()
    print("Column Names and Types:\n")
    print(df.dtypes.to_string())
    print()
    print("---------------------------------------------------")
    print()
    print("Basic Statistics:\n")
    for c in df.columns:
        print(c)
        print()
        print(df.describe()[c].to_string())
        print()


def visualize_corr_matrix(df, cmap=sns.diverging_palette(220, 10, as_cmap=True)):
	corr_matrix = df.corr(method='pearson') 

	fig, ax = plt.subplots(figsize=(10, 8))
	sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), cmap=cmap, 
            square=True, ax=ax)

# def find_nulls()

def get_correlated_features(df, target, threshold=0.2):
    corr_matrix = df.corr()
    print(f"Correlations with {target}:\n")
    correlations = corr_matrix[target].drop(target)
    print(correlations.to_string())
    print()
    print(f"Optimal features based on absolute threshold: {threshold}")
    print()
    abs_corrs = correlations.abs()
    high_corrs = abs_corrs > threshold
    subset = abs_corrs[high_corrs]
    print(subset.to_string())



