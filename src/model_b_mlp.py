
# src/model_b_mlp.py
# ------------------------------------------------------
# Modelo B: MLP (Pipeline con StandardScaler)
# Incluye:
# - Entrenamiento base
# - Evaluación
# - GridSearchCV opcional
# ------------------------------------------------------

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


# ======================================================
# 1) Pipeline base
# ======================================================
def construir_mlp_pipeline(random_state=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=5e-4,
            batch_size=256,
            learning_rate="adaptive",
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=random_state,
            verbose=False
        ))
    ])


# ======================================================
# 2) Entrenamiento base
# ======================================================
def entrenar_mlp(X_train, y_train, random_state=42):
    model = construir_mlp_pipeline(random_state=random_state)
    model.fit(X_train, y_train)
    return model


# ======================================================
# 3) Evaluación
# ======================================================
def evaluar_mlp(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, digits=4)

    print("Accuracy (MLP):", acc)
    print("ROC AUC (MLP):", auc)
    print("\nClassification Report (MLP):\n", report)

    return {
        "accuracy": acc,
        "roc_auc": auc,
        "report": report,
        "y_pred": y_pred,
        "y_proba": y_proba
    }


# ======================================================
# 4) GridSearchCV (OPCIONAL)
# ======================================================
def gridsearch_mlp(X, y, cv, random_state=42, n_jobs=-1, verbose=1):
    """
    Ejecuta GridSearchCV para el MLP usando ROC-AUC como métrica.
    Retorna el mejor estimador y el objeto grid completo.
    """

    pipe = construir_mlp_pipeline(random_state=random_state)

    param_grid = {
        "mlp__hidden_layer_sizes": [(64, 32), (128, 64), (64, 32, 16)],
        "mlp__alpha": [1e-4, 5e-4, 1e-3],
        "mlp__batch_size": [128, 256],
        "mlp__learning_rate": ["constant", "adaptive"]
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose
    )

    grid.fit(X, y)

    print("\n✅ GridSearchCV MLP completado")
    print("Mejores hiperparámetros:", grid.best_params_)
    print("Mejor ROC-AUC (CV):", round(grid.best_score_, 4))

    return grid.best_estimator_, grid
