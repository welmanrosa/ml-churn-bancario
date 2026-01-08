# src/cv_utils.py
# --------------------------------------------------------------------------------
# Utilidades para Validación Cruzada (CV) y curvas OOF
# --------------------------------------------------------------------------------
import os
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
# Utilidades IO
# ======================================================
def _ensure_dirs(base: str = "reports") -> str:
    """
    Crea la carpeta para figuras si no existe.
    Retorna la ruta reports/figures
    """
    fig_dir = os.path.join(base, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir

def _save_fig(path: str):
    """
    Guarda la figura actual y la cierra.
    """
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

# ======================================================
# 1) Crear esquema de validación cruzada (estratificada)
# ======================================================
def crear_cv(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    """
    Crea un StratifiedKFold reproducible.
    """
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

# ======================================================
# 2) Reporte de métricas en CV
# ======================================================
def reporte_cv(nombre_modelo: str, estimador, X, y, cv: StratifiedKFold):
    """
    Imprime métricas de validación cruzada (media ± std).
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
    for metric in [
        'test_accuracy',
        'test_roc_auc',
        'test_f1',
        'test_precision',
        'test_recall'
    ]:
        vals = resultados[metric]
        print(f"{metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    return resultados

# ======================================================
# 3) Probabilidades OOF
# ======================================================
def obtener_probas_oof(
    modelos: dict,
    X,
    y,
    cv: StratifiedKFold,
    proba_index: int = 1
) -> dict:
    """
    Obtiene probabilidades out-of-fold (OOF) para múltiples modelos.
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
# 4) Curvas ROC OOF
# ======================================================
def plot_roc_oof(
    y,
    y_proba_dict: dict,
    title: str = "Curva ROC – Validación cruzada (OOF)",
    reports_dir: str = "reports"
):
    fig_dir = _ensure_dirs(reports_dir)
    plt.figure(figsize=(8, 6))
    for name, y_proba in y_proba_dict.items():
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Aleatorio')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    _save_fig(os.path.join(fig_dir, "roc_oof.png"))

# ======================================================
# 5) Curvas Precision–Recall OOF
# ======================================================
def plot_pr_oof(
    y,
    y_proba_dict: dict,
    title: str = "Curva Precision–Recall – Validación cruzada (OOF)",
    reports_dir: str = "reports"
):
    fig_dir = _ensure_dirs(reports_dir)
    pos_rate = np.mean(y)
    plt.figure(figsize=(8, 6))
    for name, y_proba in y_proba_dict.items():
        prec, rec, _ = precision_recall_curve(y, y_proba)
        ap = average_precision_score(y, y_proba)
        plt.plot(rec, prec, lw=2, label=f"{name} (PR-AUC={ap:.3f})")
    plt.hlines(
        pos_rate,
        xmin=0,
        xmax=1,
        linestyles='--',
        colors='gray',
        label=f"Baseline (positivos={pos_rate:.2f})"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    _save_fig(os.path.join(fig_dir, "pr_oof.png"))

# ======================================================
# 6) Selección de umbral óptimo
# ======================================================
def seleccionar_umbral(
    y_true,
    y_proba,
    objetivo: str = "f1",
    min_precision=None,
    min_recall=None
):
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    if objetivo == "f1":
        f1_vals = 2 * prec * rec / (prec + rec + 1e-12)
        f1_thr = f1_vals[1:]
        # Máscara de restricciones (si se usan)
        mask = np.ones_like(f1_thr, dtype=bool)
        if min_precision is not None:
            mask &= (prec[1:] >= float(min_precision))
        if min_recall is not None:
            mask &= (rec[1:] >= float(min_recall))
        if not np.any(mask):
            idx = int(np.nanargmax(f1_thr))
        else:
            idxs = np.where(mask)[0]
            idx = idxs[int(np.nanargmax(f1_thr[idxs]))]
        resumen = {
            "precision": float(prec[idx + 1]),
            "recall": float(rec[idx + 1]),
            "f1": float(f1_vals[idx + 1]),
        }
        metrics_plot = {
            "thresholds": thr,
            "metric_vals": f1_thr,
            "label": "F1"
        }
        return thr[idx], resumen, metrics_plot
    raise ValueError("Solo se implementa selección por F1 en esta versión.")

def evaluar_umbral(y_true, y_proba, thr: float):
    y_pred_thr = (y_proba >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred_thr)
    report = classification_report(y_true, y_pred_thr, digits=4)
    return y_pred_thr, cm, report

# ======================================================
# 7) Métrica vs Umbral
# ======================================================
def plot_metric_vs_thresholds(
    metrics_plot: dict,
    reports_dir: str = "reports"
):
    thr = metrics_plot.get("thresholds")
    vals = metrics_plot.get("metric_vals")
    label = metrics_plot.get("label", "Métrica")
    if thr is None or vals is None:
        print("[Aviso] metrics_plot incompleto.")
        return
    fig_dir = _ensure_dirs(reports_dir)
    plt.figure(figsize=(8, 5))
    plt.plot(thr, vals, lw=2)
    plt.xlabel("Umbral")
    plt.ylabel(label)
    plt.title(f"{label} vs Umbral")
    plt.grid(alpha=0.3)
    fname = f"{label.lower()}_vs_threshold.png"
    _save_fig(os.path.join(fig_dir, fname))

# --------------------------------------------------------------------------------
# --- NUEVO: Matriz de confusión (CSV + PNG) sin crear archivos adicionales ---
# --------------------------------------------------------------------------------
import seaborn as sns
import pandas as pd

def save_confusion_matrix(cm, labels, model_name: str, reports_dir: str = "reports"):
    """
    Guarda la matriz de confusión como CSV (reports/metrics) y PNG (reports/figures).
    """
    fig_dir = _ensure_dirs(reports_dir)
    met_dir = os.path.join(reports_dir, "metrics")
    os.makedirs(met_dir, exist_ok=True)

    # CSV
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.to_csv(os.path.join(met_dir, f"cm_{model_name}.csv"), index=True)

    # PNG
    plt.figure(figsize=(4.8, 4.2))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Matriz de Confusión — {model_name}")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.tight_layout()
    _save_fig(os.path.join(fig_dir, f"cm_{model_name}.png"))