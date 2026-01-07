# src/eda.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set(style="whitegrid", color_codes=True)

def ejecutar_eda(dataset):
    print("\nDistribución del objetivo:")
    print(dataset['attrition_flag'].value_counts(normalize=True))

    num_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()

    if {'customer_age','attrition_flag'}.issubset(dataset.columns):
        sns.histplot(dataset, x='customer_age', hue='attrition_flag', bins=30, multiple='stack')
        plt.title("Edad vs Attrition")
        plt.show()

    if len(num_cols) > 1:
        corr = dataset[num_cols].corr()
        sns.heatmap(corr, cmap="YlGnBu")
        plt.title("Correlaciones numéricas")
        plt.show()
``
