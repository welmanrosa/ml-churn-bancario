
# src/cv_utils.py
# ------------------------------------------------------
# Utilidades para Validación Cruzada (CV) y curvas OOF
# ------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    cross_val_predict
)

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ======================================================
# 1) Crear esquema de validación cruzada (estratificada)
# ======================================================
def crear_cv(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    """
    Crea un StratifiedKFold reproducible.

    Parámetros
    ----------
    n_splits : int
        Número de folds (por defecto 5).
    random_state : int
        Semilla para reproducibilidad.

    Retorna
    -------
    StratifiedKFold
        Objeto de CV estratificado con shuffle=True.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# ======================================================
# 2) Reporte de métricas en CV (tu función original)
# ======================================================
def reporte_cv(nombre_modelo: str, estimador, X, y, cv: StratifiedKFold):
    """
    Imprime métricas de validación cruzada (media ± std) para un estimador dado.

    Parámetros
    ----------
    nombre_modelo : str
        Nombre legible del modelo (para el encabezado).
    estimador : sklearn.base.BaseEstimator
        Modelo scikit-learn ya configurado (no entrenado).
    X : pd.DataFrame o np.ndarray
        Variables predictoras.
    y : pd.Series o np.ndarray
        Variable objetivo binaria (0/1).
    cv : StratifiedKFold
        Esquema de validación cruzada estratificada.

    Retorna
    -------
    dict[str, np.ndarray]
        Diccionario con arrays de métricas por fold.
    """
    resultados = cross_validate(
        estimator=estimador,
        X=X,
        y=y,
        cv=cv,
        scoring=['accuracy', 'roc_auc', 'f1', 'precision', 'recall'],
        return_train_score=False,
        n_jobs=-1
    )

    print(f"\n=== Validación cruzada ({cv.n_splits} folds) – {nombre_modelo} ===")
    for metric in ['test_accuracy', 'test_roc_auc', 'test_f1', 'test_precision', 'test_recall']:
        vals = resultados[metric]
        print(f"{metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    return resultados


# ======================================================
# 3) Probabilidades OOF para uno o varios modelos
# ======================================================
def obtener_probas_oof(modelos: dict, X, y, cv: StratifiedKFold, proba_index: int = 1) -> dict:
    """
    Obtiene probabilidades out-of-fold (OOF) para uno o varios modelos.

    Parámetros
    ----------
    modelos : dict[str, sklearn.base.BaseEstimator]
        Diccionario {nombre_modelo: estimador}. Los estimadores deben soportar predict_proba.
    X, y : datos
        Matrices de entrenamiento y vector objetivo.
    cv : StratifiedKFold
        Esquema de validación cruzada.
    proba_index : int
        Índice de la columna de probas positiva (por defecto 1).

    Retorna
    -------
    dict[str, np.ndarray]
        Diccionario {nombre_modelo: y_proba_oof}.
    """
    y_proba_dict = {}
    for name, est in modelos.items():
        y_proba = cross_val_predict(
            estimator=est,
            X=X,
            y=y,
            cv=cv,
            method='predict_proba',
            n_jobs=-1
        )[:, proba_index]
        y_proba_dict[name] = y_proba
    return y_proba_dict


# ======================================================
# 4) Curvas ROC comparativas (OOF)
# ======================================================
def plot_roc_oof(y, y_proba_dict: dict, title: str = 'Curva ROC – Validación cruzada (OOF)'):
    """
    Dibuja curvas ROC usando probabilidades OOF para múltiples modelos.

    Parámetros
    ----------
    y : array-like
        Etiquetas verdaderas binaras (0/1).
    y_proba_dict : dict[str, np.ndarray]
        {nombre_modelo: proba_positiva_oof}.
    title : str
        Título del gráfico.
    """
    plt.figure(figsize=(8, 6))

    for name, y_proba in y_proba_dict.items():
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc:.3f})')

    plt.plot([0, 1], [0, 1], '--', color='gray', label='Aleatorio')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ======================================================
# 5) Curvas Precision–Recall comparativas (OOF)
# ======================================================
def plot_pr_oof(y, y_proba_dict: dict, title: str = 'Curva Precision–Recall – Validación cruzada (OOF)'):
    """
    Dibuja curvas Precision–Recall usando probabilidades OOF para múltiples modelos.

    Parámetros
    ----------
    y : array-like
        Etiquetas verdaderas binaras (0/1).
    y_proba_dict : dict[str, np.ndarray]
        {nombre_modelo: proba_positiva_oof}.
    title : str
        Título del gráfico.
    """
    pos_rate = np.mean(y)
    plt.figure(figsize=(8, 6))

    for name, y_proba in y_proba_dict.items():
        prec, rec, _ = precision_recall_curve(y, y_proba)
        ap = average_precision_score(y, y_proba)
        plt.plot(rec, prec, lw=2, label=f'{name} (PR-AUC={ap:.3f})')

    plt.hlines(
        pos_rate, xmin=0, xmax=1,
        linestyles='--', colors='gray',
        label=f'Baseline (positivos={pos_rate:.2f})'
    )

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ======================================================
# 6) Selección de umbral óptimo (F1 / Precision / Recall)
#    Basado en tu lógica original, encapsulado en funciones
# ======================================================
def seleccionar_umbral(y_true, y_proba, objetivo='f1', min_precision=None, min_recall=None):
    """
    Selecciona un umbral óptimo según objetivo:
    - 'f1'       : maximiza F1
    - 'recall'   : maximiza Recall, con opcional min_precision
    - 'precision': maximiza Precision, con opcional min_recall

    Retorna
    -------
    thr_star : float
        Umbral seleccionado.
    resumen : dict
        Métricas en el umbral (precision, recall, f1 cuando aplica).
    metrics_plot : dict
        Estructura para graficar la métrica vs thresholds.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    metrics_plot = {}

    if objetivo == 'f1':
        f1_vals = 2 * prec * rec / (prec + rec + 1e-12)
        f1_thr = f1_vals[1:]  # alineado a thresholds
        idx = int(np.nanargmax(f1_thr))
        thr_star = thr[idx]
        metrics_plot = {'thresholds': thr, 'metric_vals': f1_thr, 'label': 'F1'}
        resumen = {
            'precision': float(prec[idx + 1]),
            'recall': float(rec[idx + 1]),
            'f1': float(f1_vals[idx + 1])
        }
        return thr_star, resumen, metrics_plot

    elif objetivo == 'recall':
        candidates = []
        for i in range(len(thr)):
            p = prec[i + 1]; r = rec[i + 1]
            if (min_precision is None) or (p >= min_precision):
                candidates.append((thr[i], p, r))

        if not candidates:
            i_best = int(np.nanargmax(rec[1:]))
            thr_star = thr[i_best]
            resumen = {'precision': float(prec[i_best + 1]), 'recall': float(rec[i_best + 1])}
            metrics_plot = {'thresholds': thr, 'metric_vals': rec[1:], 'label': 'Recall'}
            return thr_star, resumen, metrics_plot

        thr_star, p_best, r_best = max(candidates, key=lambda t: t[2])
        resumen = {'precision': float(p_best), 'recall': float(r_best)}
        metrics_plot = {'thresholds': thr, 'metric_vals': rec[1:], 'label': 'Recall'}
        return thr_star, resumen, metrics_plot

    elif objetivo == 'precision':
        candidates = []
        for i in range(len(thr)):
            p = prec[i + 1]; r = rec[i + 1]
            if (min_recall is None) or (r >= min_recall):
                candidates.append((thr[i], p, r))

        if not candidates:
            i_best = int(np.nanargmax(prec[1:]))
            thr_star = thr[i_best]
            resumen = {'precision': float(prec[i_best + 1]), 'recall': float(rec[i_best + 1])}
            metrics_plot = {'thresholds': thr, 'metric_vals': prec[1:], 'label': 'Precision'}
            return thr_star, resumen, metrics_plot

        thr_star, p_best, r_best = max(candidates, key=lambda t: t[1])
        resumen = {'precision': float(p_best), 'recall': float(r_best)}
        metrics_plot = {'thresholds': thr, 'metric_vals': prec[1:], 'label': 'Precision'}
        return thr_star, resumen, metrics_plot

    else:
        raise ValueError("objetivo debe ser 'f1', 'recall' o 'precision'.")


def evaluar_umbral(y_true, y_proba, thr: float):
    """
    Evalúa un umbral dado: genera predicciones binarias y reporta
    matriz de confusión y classification_report.

    Retorna
    -------
    y_pred_thr : np.ndarray
        Predicciones binarias con el umbral.
    cm : np.ndarray
        Matriz de confusión (tn, fp, fn, tp).
    report : str
        Classification report como texto.
    """
    y_pred_thr = (y_proba >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred_thr)
    report = classification_report(y_true, y_pred_thr, digits=4)
    return y_pred_thr, cm, report


# ======================================================
# 7) Plot auxiliar para métrica vs umbrales
# ======================================================
def plot_metric_vs_thresholds(metrics_plot: dict):
    """
    Grafica la métrica (F1/Recall/Precision) en función de los thresholds.

    Parámetros
    ----------
    metrics_plot : dict
        {'thresholds': np.ndarray,
         'metric_vals': np.ndarray,
         'label': str}
    """
    thr = metrics_plot.get('thresholds')
    vals = metrics_plot.get('metric_vals')
    label = metrics_plot.get('label', 'Métrica')

    if thr is None or vals is None:
        print("[Aviso] metrics_plot no contiene datos suficientes para graficar.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(thr, vals, lw=2)
    plt.xlabel('Umbral')
    plt.ylabel(label)
    plt.title(f'{label} vs Umbral')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
