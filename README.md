
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


