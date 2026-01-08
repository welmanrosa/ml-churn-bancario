
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

## 📦 Metodologia para Solución.
### Paso 1. Definición del Problema, datos y variables.

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

### Paso 2. Evaluar si el problema es de regresión, clasificación, clusterización, o predicción de series temporales.
- Explicar cuales son las variables de entrada y salida.

a)  Tipo de problema

✔ Clasificación supervisada binaria

Clases:

0: Cliente activo
1: Cliente que abandonó (churn)



b) Variables de entrada (X)

Características socioeconómicas y de comportamiento del cliente.

c) Variable de salida (y)

Attrition_Flag (binaria).

- ¿Es posible resolver dicho problema de forma eficiente sin recurrir a inteligencia artificial?

La pregunta es: ¿Puede resolverse sin Machine Learning?
No de forma eficiente, porque:

La relación entre variables no es lineal.
Existen interacciones complejas entre comportamiento transaccional y churn.
Reglas manuales resultarían rígidas y poco escalables.

He aquí este razonamiento el cual justifica el uso de ML frente a reglas heurísticas tradicionales.

### Paso 3. Limpieza y Transformación de Datos.

Proceso de desarrollo para este Paso 3 es:

✔ Limpieza

Lectura correcta del CSV (;).
Nombres de columnas normalizados.
Eliminación de columnas irrelevantes (Naive Bayes).

✔ Codificación

One-Hot Encoding para variables categóricas:

gender, education_level, marital_status, etc.


✔ Datos desbalanceados

Distribución:

≈ 84% activos
≈ 16% churn


Se manejó mediante:

Métricas adecuadas (ROC‑AUC, PR‑AUC).
Umbral óptimo en lugar de accuracy puro.


✔ (Opcional) Normalización

Se aplicó solo cuando el modelo lo requiere, en este caso esencialmente para el Modelo B (MLP).
Uso correcto de StandardScaler dentro de un Pipeline.

✔ Partición del dataset

train_test_split estratificado (80% / 20%).

### Paso 4. Entrenamiento de Modelos.
- Redes neuronales MLPRegressor, Máquinas de soporte vectorial SVM, Árboles de decisión DecisionTree, Bosques aleatorios RandomForest, o Métodos de ensamble AdaBoost y GradientBoosting.

Se desarrollaron dos Modelos:

1 - Modelo A: Regresión Logística

Modelo lineal, interpretable.
Funciona como baseline.
Permite analizar coeficientes y efectos marginales.

2 - Modelo B: Red Neuronal (MLPClassifier)

Modelo no lineal.
Captura interacciones complejas.
Mejor rendimiento en ROC‑AUC y PR‑AUC.

Se realizó el siguiente proceso de ciencia de datos en la implementación de Machine Learning.  
- Modelo A: Regresión Logística
Definición del modelo
Entrenamiento
Evaluación inicial (train/test)
Curva ROC
Matriz de confusión
Interpretabilidad (coeficientes)

- Modelo B: Red Neuronal (MLP)
Aquí entra todo lo que se ha desarrollado desde la parte 1 hasta la 8 del Modelo B.

Pipeline (StandardScaler + MLP)
Arquitectura
Entrenamiento
Evaluación en test
Curvas ROC / PR

### Paso 5. [Model tuning]

- Utilizar un enfoque de train-validation-test o validación cruzada en lugar de train-test.

Aquí NO se crea un nuevo modelo, sino que:

Se evalúan variantes A vs B de forma rigurosa.

Aquí entra lo que se ha desarrollado en la parte 5 del Modelo A y de la parte 5 a la 8 del Modelo B.

Validación cruzada estratificada (k=5)
Métricas múltiples
Comparación estabilidad A vs B
(Opcional) GridSearch para MLP

Algo muy importante es:

El tuning no se hace en abstracto
Se hace para decidir entre modelo

### Paso 6: [Output] Análisis y Conclusión.

- Si su problema es de clasificación, proveer matriz de confusión.

- Si su problema es de regresión, proveer gráfico de salidas reales vs. predichas.

Para Seleccionar el mejor modelo se da respuesta a:

Este paso NO es “otro modelo”, es: Una decisión basada en evidencia.

Qué entra aquí para tomar la decisión:

Se analizan:
Tabla comparativa final
Métricas TEST y CV
Curvas ROC / PR
Matrices de confusión
Umbral óptimo
Justificación final

“¿Cuál modelo es mejor y por qué?”

Del análisis realizado:

Se selecciona MLP
No por gusto, sino por lo siguiente:

Mejor ROC-AUC
Mejor Recall en churn
Desempeño consistente en CV

Esto se puede visualizar y analizar en la parte 8 y 9 del Modelo B, donde se comparan los dos modelos paralelamente. 

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








