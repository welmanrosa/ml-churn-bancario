
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

## ⚙️ Proceso de Ejecución

```bash
a) Clonar el repositorio.
# Reemplaza USUARIO por tu usuario real de GitHub
git clone https://github.com/USUARIO/ml-churn-bancario.git
cd ml-churn-bancario

b) Crear entorno virtual.

# Linux / macOS
python3 -m venv venv

# Windows (PowerShell)
python -m venv venv

c) Activar el entorno virtual.

# Linux / macOS
source venv/bin/activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Nota (Windows): si PowerShell bloquea la activación del entorno virtual, ejecuta:
# (ejecuta esto en una consola de PowerShell con privilegios de usuario)
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
# Luego vuelve a activar:
.\venv\Scripts\Activate.ps1

d) Actualizar pip.
pip install --upgrade pip

e) Instalar dependencias del proyecto.
pip install -r requirements.txt

f) Ejecutar el proyecto.
python main.py --save-reports

# (Opcional) Uso con Conda en lugar de venv:
# Crear y activar entorno
conda create -n churn-env python=3.10 -y
conda activate churn-env
# Instalar dependencias y ejecutar
pip install -r requirements.txt
python main.py --save-reports


