# Churn Bancario – Comparación LogReg vs MLP

Proyecto de Machine Learning aplicado a datos bancarios públicos,
enfocado en el problema de fuga de clientes (churn / attrition).

Se realiza una comparación entre:
- **Modelo A**: Regresión Logística
- **Modelo B**: Red Neuronal Multicapa (MLP)

El proyecto incluye análisis exploratorio de datos (EDA),
ingeniería de variables, validación cruzada estratificada (OOF),
curvas ROC y Precision–Recall, y selección de umbral óptimo.

---

## 📦 Dataset
- Dataset público de clientes bancarios
- Variable objetivo: `attrition_flag` (0 = Existing, 1 = Attrited)
- Carga desde URL `raw` de GitHub

---

## ⚙️ Requisitos
```bash
pip install -r requirements.txt
