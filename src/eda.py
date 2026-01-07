
# src/eda.py
"""
Análisis Exploratorio de Datos (EDA) para churn bancario.

- Imprime distribución del objetivo (attrition_flag)
- Genera histograma de edad por clase (si existen las columnas)
- Genera mapa de calor de correlaciones (numéricas)
- Guarda las figuras en reports/figures/
- No usa plt.show() (compatible con backend Agg / ejecución headless)
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Estilo consistente para las figuras
sns.set(style="whitegrid", color_codes=True)


def _ensure_dirs(base: str = "reports") -> str:
    """
    Crea la carpeta de figuras si no existe.
    Retorna la ruta: reports/figures
    """
    fig_dir = os.path.join(base, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def ejecutar_eda(dataset: pd.DataFrame, reports_dir: str = "reports") -> None:
    """
    Ejecuta EDA básico sobre el dataset:
    - Distribución del objetivo
    - Histograma de edad por clase (si aplica)
    - Heatmap de correlaciones numéricas

    Parameters
    ----------
    dataset : pd.DataFrame
        Datos con columnas como 'attrition_flag', 'customer_age', etc.
    reports_dir : str
        Directorio base donde se guardarán las figuras ('reports/figures').
    """
    # 1) Distribución del objetivo
    print("\nDistribución del objetivo:")
    if "attrition_flag" in dataset.columns:
        dist = dataset["attrition_flag"].value_counts(normalize=True)
        print(dist)
    else:
        print("⚠️ 'attrition_flag' no está en las columnas del dataset.")

    # Crear carpeta de figuras
    fig_dir = _ensure_dirs(reports_dir)

    # 2) Histograma de edad por clase (si existen columnas)
    if {"customer_age", "attrition_flag"}.issubset(dataset.columns):
        try:
            plt.figure(figsize=(8, 4))
            sns.histplot(
                data=dataset,
                x="customer_age",
                hue="attrition_flag",
                bins=30,
                multiple="stack",
                stat="count",
                edgecolor="black",
            )
            plt.title("Distribución de Edad por Attrition")
            plt.xlabel("Edad del cliente")
            plt.ylabel("Frecuencia")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "eda_age_vs_attrition.png"), dpi=140)
            plt.close()
        except Exception as e:
            print(f"[EDA] Error generando histograma edad vs attrition: {e}")
    else:
        print("ℹ️ No se generan histogramas: faltan columnas {'customer_age','attrition_flag'}.")

    # 3) Heatmap de correlaciones numéricas
    num_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 1:  # ← corregido: '>' en lugar de '&gt;'
        try:
            corr = dataset[num_cols].corr(numeric_only=True)
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                corr,
                cmap="YlGnBu",
                annot=False,
                fmt=".2f",
                cbar=True,
                square=False,
            )
            plt.title("Correlaciones (columnas numéricas)")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "eda_corr_heatmap.png"), dpi=140)
            plt.close()
        except Exception as e:
            print(f"[EDA] Error generando heatmap de correlaciones: {e}")
    else:
        print("ℹ️ No se genera heatmap: no hay suficientes columnas numéricas.")

