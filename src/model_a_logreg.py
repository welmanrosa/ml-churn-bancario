
# src/model_a_logreg.py
# ------------------------------------------------------
# Modelo A: Regresión Logística
# ------------------------------------------------------

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

def entrenar_logreg(X_train, y_train, random_state=42):
    """
    Entrena un modelo base de Regresión Logística.
    """
    model = LogisticRegression(
        solver='lbfgs',
        max_iter=500,
        random_state=random_state
    )
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

    print("Accuracy (LogReg):", acc)
    print("ROC‑AUC (LogReg):", auc)
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

