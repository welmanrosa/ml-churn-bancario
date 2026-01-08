
# main.py
# ------------------------------------------------------------------------------
# Orquestador del proyecto:
# - Carga y limpieza de datos
# - EDA (opcional)
# - Ingeniería de variables
# - Modelo A: Regresión Logística
# - Modelo B: MLP (base o GridSearchCV)
# - Evaluación TEST
# - Validación cruzada OOF
# - Curvas ROC / PR
# - Selección de umbral óptimo (F1)
# - Exportación de resultados
# - Scores por cliente
# - Champion vs Challenger
# ------------------------------------------------------------------------------

import os
import sys
import time
import json
import argparse
import logging
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# Ajuste: silenciar SOLO el mensaje informativo de Matplotlib
# (proviene del logger, no de warnings)
# ------------------------------------------------------------------------------
import warnings
warnings.filterwarnings(
    "ignore",
    message="Using categorical units to plot a list of strings"
)

# Backend seguro para matplotlib (CI/CD, servidores)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Silenciar logs INFO de Matplotlib
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.category").setLevel(logging.ERROR)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
    confusion_matrix, classification_report
)

# =========================
# Imports del paquete src
# =========================
from src import (
    cargar_dataset,
    ejecutar_eda,
    preparar_features,
    entrenar_logreg,
    evaluar_logreg,
    crear_cv,
    reporte_cv,
    obtener_probas_oof,
    plot_roc_oof,
    plot_pr_oof,
    seleccionar_umbral,
    evaluar_umbral,
    plot_metric_vs_thresholds
)

from src.model_b_mlp import (
    entrenar_mlp,
    evaluar_mlp,
    gridsearch_mlp
)

# Utilidades añadidas (sin archivos nuevos)
from src.cv_utils import save_confusion_matrix
from src.eda import (
    eda_categoricas_vs_clase,
    eda_continuas_vs_clase,
    eda_scatter_clase_vs_var,
    eda_scatter_pares_continuas
)

# =========================
# Logging
# =========================
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s\n%(levelname)s\n%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

# =========================
# CLI arguments
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Churn bancario – LogReg vs MLP"
    )
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--save-reports", action="store_true")
    parser.add_argument(
        "--do-grid",
        action="store_true",
        help="Ejecuta GridSearchCV para el MLP"
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()

# =========================
# Utilidades IO
# =========================
def ensure_dirs(base="reports"):
    fig_dir = os.path.join(base, "figures")
    met_dir = os.path.join(base, "metrics")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(met_dir, exist_ok=True)
    return fig_dir, met_dir

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

# =========================
# Helpers: impresión de matrices de confusión
# =========================
def _df_cm(cm: np.ndarray):
    labels = ["Existing(0)", "Attrited(1)"]
    return pd.DataFrame(
        cm,
        index=pd.Index(labels, name="Real"),
        columns=pd.Index(labels, name="Predicho")
    )

def print_confusion_matrix(cm: np.ndarray, model_name: str):
    """
    Imprime la matriz de confusión (conteos absolutos) con etiquetas legibles.
    cm: array 2x2 (TN, FP / FN, TP)
    """
    if cm.shape != (2, 2):
        logging.info("Matriz de confusión (%s) no es 2x2: shape=%s", model_name, cm.shape)
        logging.info("\n%s", cm)
        return
    df_cm = _df_cm(cm)
    logging.info("\n=== MATRIZ DE CONFUSIÓN — %s (conteos) ===\n%s", model_name, df_cm.to_string())

def print_confusion_matrix_percent(cm: np.ndarray, model_name: str):
    """
    Imprime la matriz de confusión en porcentajes sobre el total de muestras (TEST).
    """
    if cm.shape != (2, 2):
        return
    total = cm.sum()
    if total == 0:
        logging.info("\n=== MATRIZ DE CONFUSIÓN — %s (porcentajes) ===\nSin datos.", model_name)
        return
    df_pct = _df_cm(np.round(cm / total * 100.0, 2))
    logging.info("\n=== MATRIZ DE CONFUSIÓN — %s (porcentajes del total) ===\n%s", model_name, df_pct.to_string())

# =========================
# MAIN
# =========================
def main():
    setup_logging()
    args = parse_args()
    np.random.seed(args.random_state)

    logging.info("Parámetros de ejecución: %s", vars(args))
    t0 = time.time()

    # --------------------------------------------------------------------------
    # 1) Dataset
    # --------------------------------------------------------------------------
    dataset = cargar_dataset()
    logging.info("Dataset cargado. Shape: %s", dataset.shape)

    if not args.skip_eda:
        try:
            # EDA básico
            ejecutar_eda(dataset)
            # EDA ampliado
            eda_categoricas_vs_clase(dataset)
            eda_continuas_vs_clase(dataset)
            eda_scatter_clase_vs_var(dataset)
            eda_scatter_pares_continuas(dataset)
        except Exception as e:
            logging.warning("EDA omitida (%s)", e)

    # --------------------------------------------------------------------------
    # 2) Features
    # --------------------------------------------------------------------------
    X, y = preparar_features(dataset)
    logging.info("Features listos. X=%s | y=%s", X.shape, y.shape)

    # --------------------------------------------------------------------------
    # 3) Split
    # --------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=args.random_state
    )

    # --------------------------------------------------------------------------
    # 4) MODELO A — LogReg
    # --------------------------------------------------------------------------
    logging.info("Entrenando Modelo A (LogReg)...")
    model_a = entrenar_logreg(X_train, y_train)
    evaluar_logreg(model_a, X_test, y_test)

    y_pred_a = model_a.predict(X_test)
    y_proba_a = model_a.predict_proba(X_test)[:, 1]

    metrics_a = {
        "accuracy": accuracy_score(y_test, y_pred_a),
        "roc_auc": roc_auc_score(y_test, y_proba_a),
        "f1": f1_score(y_test, y_pred_a),
        "precision": precision_score(y_test, y_pred_a),
        "recall": recall_score(y_test, y_pred_a),
        "cm": confusion_matrix(y_test, y_pred_a),
        "report": classification_report(y_test, y_pred_a, digits=4)
    }

    # --- NUEVO: imprimir matrices de confusión LogReg (conteos y porcentajes) ---
    print_confusion_matrix(metrics_a["cm"], "LogReg")
    print_confusion_matrix_percent(metrics_a["cm"], "LogReg")

    # --------------------------------------------------------------------------
    # 5) MODELO B — MLP
    # --------------------------------------------------------------------------
    logging.info("Entrenando Modelo B (MLP)...")
    cv = crear_cv(
        n_splits=args.n_splits,
        random_state=args.random_state
    )

    if args.do_grid:
        logging.info("GridSearchCV activado para MLP...")
        model_b, _ = gridsearch_mlp(X, y, cv, random_state=args.random_state)
    else:
        model_b = entrenar_mlp(X_train, y_train, random_state=args.random_state)

    metrics_b = evaluar_mlp(model_b, X_test, y_test)
    y_pred_b = metrics_b["y_pred"]
    y_proba_b = metrics_b["y_proba"]

    metrics_b.update({
        "f1": f1_score(y_test, y_pred_b),
        "precision": precision_score(y_test, y_pred_b),
        "recall": recall_score(y_test, y_pred_b),
        "cm": confusion_matrix(y_test, y_pred_b)
    })

    # --- NUEVO: imprimir matrices de confusión MLP (conteos y porcentajes) ---
    print_confusion_matrix(metrics_b["cm"], "MLP")
    print_confusion_matrix_percent(metrics_b["cm"], "MLP")

    # --------------------------------------------------------------------------
    # Matrices de confusión (CSV + PNG)
    # --------------------------------------------------------------------------
    labels = ["Existing(0)", "Attrited(1)"]
    save_confusion_matrix(metrics_a["cm"], labels, "LogReg")
    save_confusion_matrix(metrics_b["cm"], labels, "MLP")

    # --------------------------------------------------------------------------
    # 6) Tabla TEST
    # --------------------------------------------------------------------------
    df_test = pd.DataFrame([
        {
            "Modelo": "LogReg",
            **{k: metrics_a[k] for k in ["accuracy","roc_auc","f1","precision","recall"]}
        },
        {
            "Modelo": "MLP",
            **{k: metrics_b[k] for k in ["accuracy","roc_auc","f1","precision","recall"]}
        }
    ]).round(4)

    logging.info("\n=== MÉTRICAS TEST ===\n%s", df_test.to_string(index=False))

    # --------------------------------------------------------------------------
    # 7) CV + OOF
    # --------------------------------------------------------------------------
    reporte_cv("LogReg", model_a, X, y, cv)
    reporte_cv("MLP", model_b, X, y, cv)

    y_probas_oof = obtener_probas_oof(
        {"LogReg": model_a, "MLP": model_b},
        X, y, cv
    )

    plot_roc_oof(y, y_probas_oof)
    plot_pr_oof(y, y_probas_oof)

    # --------------------------------------------------------------------------
    # 8) Umbrales óptimos
    # --------------------------------------------------------------------------
    thr_a, sum_a, mplot_a = seleccionar_umbral(y, y_probas_oof["LogReg"], "f1")
    thr_b, sum_b, mplot_b = seleccionar_umbral(y, y_probas_oof["MLP"], "f1")

    logging.info("Umbral LogReg: %.4f\n%s", thr_a, sum_a)
    logging.info("Umbral MLP : %.4f\n%s", thr_b, sum_b)

    plot_metric_vs_thresholds(mplot_a)
    plot_metric_vs_thresholds(mplot_b)

    # --------------------------------------------------------------------------
    # 10) CHAMPION vs CHALLENGER (impresión SIEMPRE, exportación si --save-reports)
    # --------------------------------------------------------------------------
    # Criterio: mayor ROC-AUC; si empatan, mayor F1
    crit_a = (metrics_a["roc_auc"], metrics_a.get("f1", float("nan")))
    crit_b = (metrics_b["roc_auc"], metrics_b.get("f1", float("nan")))

    if crit_b > crit_a:
        champion_model = "MLP"
        challenger_model = "LogReg"
        champion_metrics = metrics_b
        challenger_metrics = metrics_a
    else:
        champion_model = "LogReg"
        challenger_model = "MLP"
        champion_metrics = metrics_a
        challenger_metrics = metrics_b

    logging.info(
        "🏆 Champion: %s | ROC-AUC=%.4f | F1=%.4f\n"
        "🥈 Challenger: %s | ROC-AUC=%.4f | F1=%.4f",
        champion_model,
        champion_metrics["roc_auc"],
        champion_metrics.get("f1", float("nan")),
        challenger_model,
        challenger_metrics["roc_auc"],
        challenger_metrics.get("f1", float("nan"))
    )

    # Exportar CSV del resumen solo si se pidió guardar reportes
    if args.save_reports:
        _, met_dir = ensure_dirs()
        df_champion = pd.DataFrame([
            {
                "rol": "Champion",
                "modelo": champion_model,
                "roc_auc": champion_metrics["roc_auc"],
                "f1": champion_metrics.get("f1", float("nan")),
                "recall": champion_metrics.get("recall", float("nan")),
                "precision": champion_metrics.get("precision", float("nan")),
                "criterio": "Mejor ROC-AUC; en empate, mejor F1"
            },
            {
                "rol": "Challenger",
                "modelo": challenger_model,
                "roc_auc": challenger_metrics["roc_auc"],
                "f1": challenger_metrics.get("f1", float("nan")),
                "recall": challenger_metrics.get("recall", float("nan")),
                "precision": challenger_metrics.get("precision", float("nan")),
                "criterio": "Segundo mejor bajo el mismo criterio"
            }
        ]).round(4)
        df_champion.to_csv(
            os.path.join(met_dir, "champion_challenger_summary.csv"),
            index=False
        )

    # --------------------------------------------------------------------------
    # 9) Exportación (resto de archivos)
    # --------------------------------------------------------------------------
    if args.save_reports:
        fig_dir, met_dir = ensure_dirs()

        df_test.to_csv(os.path.join(met_dir, "metrics_test.csv"), index=False)

        # Scores por cliente OOF (opcional)
        id_candidates = ["customer_id", "client_id", "id_cliente"]
        id_col = next((c for c in id_candidates if c in dataset.columns), None)
        id_series = dataset[id_col] if id_col else dataset.index.to_series().rename("row_id")

        df_scores = pd.DataFrame({
            "id_cliente": id_series.values,
            "y_true": y.values,
            "proba_logreg": y_probas_oof["LogReg"],
            "proba_mlp": y_probas_oof["MLP"],
            "pred_05_logreg": (y_probas_oof["LogReg"] >= 0.50).astype(int),
            "pred_05_mlp":    (y_probas_oof["MLP"]    >= 0.50).astype(int),
            "pred_opt_logreg": (y_probas_oof["LogReg"] >= thr_a).astype(int),
            "pred_opt_mlp":    (y_probas_oof["MLP"]    >= thr_b).astype(int),
            "thr_logreg": thr_a,
            "thr_mlp":    thr_b
        })
        df_scores.to_csv(os.path.join(met_dir, "scores_clientes_oof.csv"), index=False)

        with open(os.path.join(met_dir, "report_logreg.txt"), "w", encoding="utf-8") as f:
            f.write(metrics_a["report"])
        with open(os.path.join(met_dir, "report_mlp.txt"), "w", encoding="utf-8") as f:
            f.write(metrics_b["report"])

        with open(os.path.join(met_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

    dt = time.time() - t0
    logging.info("✅ Pipeline finalizado en %.2f segundos", dt)

if __name__ == "__main__":
    main()