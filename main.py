
# main.py
# ------------------------------------------------------
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
# ------------------------------------------------------

import os
import sys
import time
import json
import argparse
import logging
import numpy as np
import pandas as pd

# Backend seguro para matplotlib (CI/CD, servidores)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# =========================
# Logging
# =========================
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
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
    parser.add_argument("--do-grid", action="store_true",
                        help="Ejecuta GridSearchCV para el MLP")
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
# MAIN
# =========================
def main():
    setup_logging()
    args = parse_args()
    np.random.seed(args.random_state)

    logging.info("Parámetros de ejecución: %s", vars(args))
    t0 = time.time()

    # ---------------------------------------------------
    # 1) Dataset
    # ---------------------------------------------------
    dataset = cargar_dataset()
    logging.info("Dataset cargado. Shape: %s", dataset.shape)

    if not args.skip_eda:
        try:
            ejecutar_eda(dataset)
        except Exception as e:
            logging.warning("EDA omitida (%s)", e)

    # ---------------------------------------------------
    # 2) Features
    # ---------------------------------------------------
    X, y = preparar_features(dataset)
    logging.info("Features listos. X=%s | y=%s", X.shape, y.shape)

    # ---------------------------------------------------
    # 3) Split
    # ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=args.random_state
    )

    # ---------------------------------------------------
    # 4) MODELO A — LogReg
    # ---------------------------------------------------
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

    # ---------------------------------------------------
    # 5) MODELO B — MLP (base o Grid)
    # ---------------------------------------------------
    logging.info("Entrenando Modelo B (MLP)...")

    cv = crear_cv(n_splits=args.n_splits, random_state=args.random_state)

    if args.do_grid:
        logging.info("GridSearchCV activado para MLP...")
        model_b, _ = gridsearch_mlp(X, y, cv, random_state=args.random_state)
    else:
        model_b = entrenar_mlp(X_train, y_train, random_state=args.random_state)

    metrics_b = evaluar_mlp(model_b, X_test, y_test)
    y_proba_b = metrics_b["y_proba"]
    y_pred_b = metrics_b["y_pred"]

    metrics_b.update({
        "f1": f1_score(y_test, y_pred_b),
        "precision": precision_score(y_test, y_pred_b),
        "recall": recall_score(y_test, y_pred_b),
        "cm": confusion_matrix(y_test, y_pred_b)
    })

    # ---------------------------------------------------
    # 6) Tabla TEST comparativa
    # ---------------------------------------------------
    df_test = pd.DataFrame([
        {"Modelo": "LogReg", **{k: metrics_a[k] for k in ["accuracy","roc_auc","f1","precision","recall"]}},
        {"Modelo": "MLP",    **{k: metrics_b[k] for k in ["accuracy","roc_auc","f1","precision","recall"]}},
    ]).round(4)

    logging.info("\n=== MÉTRICAS TEST ===\n%s", df_test.to_string(index=False))

    # ---------------------------------------------------
    # 7) VALIDACIÓN CRUZADA + OOF
    # ---------------------------------------------------
    res_a = reporte_cv("LogReg", model_a, X, y, cv)
    res_b = reporte_cv("MLP", model_b, X, y, cv)

    y_probas_oof = obtener_probas_oof(
        {"LogReg": model_a, "MLP": model_b},
        X, y, cv
    )

    plot_roc_oof(y, y_probas_oof)
    plot_pr_oof(y, y_probas_oof)

    # ---------------------------------------------------
    # 8) Umbrales óptimos (F1)
    # ---------------------------------------------------
    thr_a, sum_a, mplot_a = seleccionar_umbral(y, y_probas_oof["LogReg"], objetivo="f1")
    thr_b, sum_b, mplot_b = seleccionar_umbral(y, y_probas_oof["MLP"], objetivo="f1")

    logging.info("Umbral LogReg: %.4f | %s", thr_a, sum_a)
    logging.info("Umbral MLP   : %.4f | %s", thr_b, sum_b)

    plot_metric_vs_thresholds(mplot_a)
    plot_metric_vs_thresholds(mplot_b)

    # ---------------------------------------------------
    # 9) Exportación
    # ---------------------------------------------------
    if args.save_reports:
        fig_dir, met_dir = ensure_dirs()

        df_test.to_csv(os.path.join(met_dir, "metrics_test.csv"), index=False)

        with open(os.path.join(met_dir, "report_logreg.txt"), "w", encoding="utf-8") as f:
            f.write(metrics_a["report"])
        with open(os.path.join(met_dir, "report_mlp.txt"), "w", encoding="utf-8") as f:
            f.write(metrics_b["report"])

        save_fig(os.path.join(fig_dir, "roc_oof.png"))
        save_fig(os.path.join(fig_dir, "pr_oof.png"))

        with open(os.path.join(met_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

        logging.info("Reportes guardados en /reports")

    dt = time.time() - t0
    logging.info("✅ Pipeline finalizado en %.2f segundos", dt)

# =========================
if __name__ == "__main__":
    main()

