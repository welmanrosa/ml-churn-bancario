
# src/model_a_logreg.py
# ------------------------------------------------------
# Modelo A: Regresión Logística (robusta y escalada)
# ------------------------------------------------------

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


def entrenar_logreg(X_train, y_train, random_state: int = 42):
    """
    Entrena un modelo de Regresión Logística usando pipeline:
    - Escalado estándar de features
    - Solver robusto (lbfgs)
    - Más iteraciones para garantizar convergencia

    Retorna un Pipeline compatible con scikit-learn.
    """
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    return model


def evaluar_logreg(model, X_test, y_test):
    """
    Evalúa el modelo en TEST.
    Imprime métricas y retorna un diccionario estándar.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    f1  = f1_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print("Accuracy (LogReg):", round(acc, 4))
    print("ROC‑AUC (LogReg):", round(auc, 4))
    print("\nClassification Report (LogReg):\n", report)

    return {
        "accuracy": acc,
        "roc_auc": auc,
        "f1": f1,
        "precision": pre,
        "recall": rec,
        "cm": cm,
        "report": report,
        "y_pred": y_pred,
        "y_proba": y_proba
    }


