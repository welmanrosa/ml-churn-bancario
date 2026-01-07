
"""
Paquete src del proyecto:
Machine Learning aplicado a churn bancario.

Incluye:
- Carga y limpieza robusta de datos
- Análisis exploratorio (EDA)
- Ingeniería de variables (One-Hot Encoding)
- Modelo A: Regresión Logística
- Utilidades de validación cruzada (OOF, ROC, PR, umbrales)
"""

__version__ = "0.1.0"
__author__ = "Welman Rosa Alvarado"
__license__ = "MIT"

# Exposición de funciones clave del paquete (opcional, pero elegante)
from .data_loader import cargar_dataset
from .features import preparar_features
from .eda import ejecutar_eda

from .model_a_logreg import entrenar_logreg, evaluar_logreg

from .cv_utils import (
    crear_cv,
    reporte_cv,
    obtener_probas_oof,
    plot_roc_oof,
    plot_pr_oof,
    seleccionar_umbral,
    evaluar_umbral,
    plot_metric_vs_thresholds
)

__all__ = [
    "cargar_dataset",
    "preparar_features",
    "ejecutar_eda",
    "entrenar_logreg",
    "evaluar_logreg",
    "crear_cv",
    "reporte_cv",
    "obtener_probas_oof",
    "plot_roc_oof",
    "plot_pr_oof",
    "seleccionar_umbral",
    "evaluar_umbral",
    "plot_metric_vs_thresholds",
]
