
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

## 📦 Descripción de archivos
- main.py: Orquestador del pipeline (data→features→model→métricas)
- src/data_loader.py: Carga de datos (CSV, SQL, etc.)
- src/eda.py: Análisis exploratorio de datos (gráficas, estadísticos)
- src/features.py: Feature engineerig
- src/model_a_logreg.py: Modelo A→Regresión Logística
- src/model_b_mlp.py: Modelo B→MLP
- src/cv_utils: Cross-validation, métricas, splits.

---


## ⚙️ Proceso de Ejecución

```bash
a) Clonar el repositorio.
# En consola de git
git clone https://github.com/welmanrosa/ml-churn-bancario.git

# En terminal Linux/Windows
cd ml-churn-bancario

b) Crear entorno virtual.

# Linux / macOS
python3 -m venv .venv

# Windows (PowerShell)
python -m venv .venv

c) Activar el entorno virtual.

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (CMD)
.\.venv\Scripts\Activate.bat

# Nota (Windows): si PowerShell bloquea la activación del entorno virtual, ejecuta:
# (ejecuta esto en una consola de PowerShell con privilegios de usuario)
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
# Luego vuelve a activar:
.\.venv\Scripts\Activate.ps1


d) Confirmamos si el entorno existe. 
# En Linux/macOS (bash/zsh): .venv/bin/
# En PowerShell: (Get-ChildItem .venv\Scripts)
# En CMD: (dir .venv\Scripts)
Tienen que estar estos archivos: activate.bat Activate.ps1 python.exe

e) Actualizar pip.
python -m pip install --upgrade pip

f) Instalar dependencias del proyecto.
pip install -r requirements.txt


g) Exportar PYTHONPATH (raíz del proyecto)

# Linux / macOS
export PYTHONPATH=$(pwd)

# Windows (PowerShell)
$env:PYTHONPATH = (Get-Location)

# Windows (CMD)
set PYTHONPATH=%CD%

h) Verificar intérprete de Python activo

# Linux / macOS
which python

# Windows (PowerShell)
where python

# Windows (CMD)
where python

i) Verificar importación del módulo src

python -c "import src; print('Import de src OK')"

j) Ejecutar el proyecto.
python main.py --save-reports

# (Opcional) Uso con Conda en lugar de .venv:
# Crear y activar entorno
conda create -n churn-env python=3.10 -y
conda activate churn-env
# Instalar dependencias y ejecutar
pip install -r requirements.txt
python main.py --save-reports







## 📦 Metodologia para Solución.
### Paso 1: Definición del Problema, datos y variables.

1- Contexto del problema: Un gerente de una entidad bancaria esta interesado en saber porque cada vez sus clientes abandonan los servicios de la tarjeta de crédito. Necesita predecir con anticipación qué cliente esta a punto de abandonar y poder así ofrecerle mejor servicio o fidelizarlo ofreciendo otro producto mejor. Se tiene un conjunto de datos consta de 10,000 instancias o clientes donde contiene información acerca del cliente como: la edad, genero, categoría salarial, estado civil, número de dependientes, tipo de tarjeta de crédito, periodo de relación con el bando, etc.
Objetivo: Predecir si un cliente dejará o abandonará los servicios de tarjeta de crédito de la entidad bancaria.

2- Descripción del dataset y variables.

✔ Dataset

BankChurners – Credit Card Customers
✔ Fuente

Repositorio personal (basado en dataset de Kaggle – Credit Card Customers).

✔ Número de observaciones

10,127 clientes

✔ Variables


Variable objetivo:

Attrition_Flag: Indica si el cliente abandonó el banco.


Variables predictoras:

Demográficas: edad, género, estado civil, ingresos.
Comportamiento: transacciones, meses inactivo, uso de crédito, etc.



✔ Tipo de datos

Mixtos: numéricos y categóricos.

Este paso está completamente cubierto en la Parte 1 del Modelo A (Carga y limpieza).

---
