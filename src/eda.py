
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
    ---------
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
            plt.title("Distribución de Edad por Attrition (0=Existing, 1=Attrited)")
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
    if len(num_cols) > 1:  # ← corregido: '>' en lugar de '>'
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

# --------------------------------------------------------------------
# --- NUEVO: EDA ampliado (sin crear archivos adicionales) ---
# --------------------------------------------------------------------

# Helpers para detectar columnas presentes
def _categoricas_presentes(dataset, candidatas):
    return [c for c in candidatas if c in dataset.columns]

def _continuas_presentes(dataset, candidatas):
    return [c for c in candidatas if c in dataset.columns]

# Categóricas vs clase (countplots)
def eda_categoricas_vs_clase(dataset: pd.DataFrame, reports_dir: str = "reports") -> None:
    fig_dir = _ensure_dirs(reports_dir)
    if "attrition_flag" not in dataset.columns:
        print("[EDA] Falta 'attrition_flag' para graficar categóricas vs clase.")
        return

    categs = _categoricas_presentes(dataset, [
        "gender", "education_level", "marital_status",
        "income_category", "card_category"
    ])
    if not categs:
        print("[EDA] No hay categóricas disponibles.")
        return

    for c in categs:
        plt.figure(figsize=(7, 4))
        sns.countplot(data=dataset, x=c, hue="attrition_flag")
        plt.title(f"{c} vs Attrition (0=Existing, 1=Attrited)")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"eda_count_{c}.png"), dpi=140)
        plt.close()

# Continuas vs clase (box + violin)
def eda_continuas_vs_clase(dataset: pd.DataFrame, reports_dir: str = "reports") -> None:
    fig_dir = _ensure_dirs(reports_dir)
    if "attrition_flag" not in dataset.columns:
        print("[EDA] Falta 'attrition_flag' para graficar continuas vs clase.")
        return

    conts = _continuas_presentes(dataset, [
        "customer_age", "credit_limit", "total_trans_amt",
        "total_trans_ct", "avg_utilization_ratio", "total_revolving_bal"
    ])
    if not conts:
        print("[EDA] No hay continuas disponibles.")
        return

    for c in conts:
        # Boxplot
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=dataset, x="attrition_flag", y=c)
        plt.title(f"Boxplot de {c} por clase (0=Existing, 1=Attrited)")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"eda_box_{c}.png"), dpi=140)
        plt.close()

        # Violin
        plt.figure(figsize=(7, 4))
        sns.violinplot(data=dataset, x="attrition_flag", y=c, inner="quartile", cut=0)
        plt.title(f"Violin de {c} por clase (0=Existing, 1=Attrited)")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"eda_violin_{c}.png"), dpi=140)
        plt.close()

# Dispersión clase vs variable (stripplot)
def eda_scatter_clase_vs_var(dataset: pd.DataFrame, reports_dir: str = "reports") -> None:
    fig_dir = _ensure_dirs(reports_dir)
    if "attrition_flag" not in dataset.columns:
        print("[EDA] Falta 'attrition_flag' para stripplot.")
        return

    conts = _continuas_presentes(dataset, [
        "customer_age", "credit_limit", "total_trans_amt",
        "avg_utilization_ratio", "total_trans_ct"
    ])
    if not conts:
        print("[EDA] No hay continuas disponibles para stripplot.")
        return

    for c in conts:
        plt.figure(figsize=(7, 4))
        sns.stripplot(data=dataset, x="attrition_flag", y=c, jitter=0.25, alpha=0.6)
        plt.title(f"Dispersión (strip) de {c} vs clase")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"eda_strip_{c}.png"), dpi=140)
        plt.close()

# Scatter de pares de continuas coloreado por clase
def eda_scatter_pares_continuas(dataset: pd.DataFrame, reports_dir: str = "reports") -> None:
    fig_dir = _ensure_dirs(reports_dir)
    if "attrition_flag" not in dataset.columns:
        print("[EDA] Falta 'attrition_flag' para scatter por clase.")
        return

    pares = [
        ("credit_limit", "avg_utilization_ratio"),
        ("total_trans_amt", "total_trans_ct"),
        ("customer_age", "total_relationship_count"),
    ]
    pares = [(x, y) for (x, y) in pares if x in dataset.columns and y in dataset.columns]
    if not pares:
        print("[EDA] No hay pares continuos disponibles.")
        return

    for x, y in pares:
        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=dataset, x=x, y=y, hue="attrition_flag", alpha=0.6, edgecolor=None)
        plt.title(f"Dispersión de {x} vs {y} por clase")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"eda_scatter_{x}_vs_{y}.png"), dpi=140)
        plt.close()
