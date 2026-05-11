# 🎓 Predicción de Abandono Escolar - ML Clásico

## 📋 Descripción del Proyecto

Este proyecto implementa un **modelo de Machine Learning clásico** para predecir el riesgo de abandono escolar de estudiantes, como parte del curso de **Artificial Intelligence Foundations** de la **Fundació URV**.

## 🎯 Formulación del Problema ML

### Tipo de Aprendizaje
- **Supervisado**: Tenemos datos etiquetados (estudiantes que abandonaron o que continuaron)

### Tarea ML
- **Clasificación Binaria**: Predecir si un estudiante abandonará (1) o continuará activo (0)

### Variable Target
- `estado`: {abandono, activo}

### Métricas de Éxito
- **Accuracy**: Porcentaje de predicciones correctas
- **Precision**: De los predichos como abandono, cuántos realmente abandonaron
- **Recall**: De los que realmente abandonaron, cuántos identificamos
- **F1-Score**: Balance entre Precision y Recall
- **ROC-AUC**: Capacidad de discriminación del modelo

---

## 📊 Características Principales

| Característica | Tipo | Descripción |
|---|---|---|
| **edad** | Numérica | Edad del estudiante (15-25 años) |
| **asistencia** | Numérica | Porcentaje de asistencia a clases (0-100%) |
| **calificacion_promedio** | Numérica | Promedio de calificaciones (0-5) |
| **horas_estudio** | Numérica | Horas dedicadas al estudio por semana (0-8) |
| **ingresos_familia** | Categórica | Nivel de ingresos familiares (bajo, medio, alto) |
| **apoyo_familiar** | Categórica | Existe apoyo familiar (si, no) |

---

## 🤖 Modelos Implementados

### 1. **Logistic Regression**
- ✅ Interpretable y rápido
- ✅ Bueno para problemas lineales
- ⚠️ Puede no capturar relaciones complejas

### 2. **Random Forest**
- ✅ Maneja datos complejos bien
- ✅ Resistente al overfitting
- ✅ Proporciona importancia de características
- ⚠️ Más lento de entrenar

### 3. **Gradient Boosting**
- ✅ Excelente rendimiento predictivo
- ✅ Captura interacciones complejas
- ⚠️ Riesgo de overfitting
- ⚠️ Más lento de entrenar

### 4. **Support Vector Machine (SVM)**
- ✅ Excelente en espacios de alta dimensión
- ✅ Versátil con diferentes kernels
- ⚠️ Menos interpretable

---

## 🚀 Instalación

### Requisitos
- Python 3.8+
- pip o conda

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/jgarola-dev/-ML-Cl-sico-Artificial-Intelligence-Foundations-Fundaci-URV-.git
cd -ML-Cl-sico-Artificial-Intelligence-Foundations-Fundaci-URV-

# 2. Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

---

## 💻 Uso

### Interfaz Web (Streamlit)

La aplicación tiene 4 secciones principales:

#### 📌 **Inicio**
- Descripción del proyecto
- Explicación del problema ML
- Información de características
- Enlaces a fuentes de datos

#### 📊 **Análisis EDA (Exploratory Data Analysis)**
- Carga tu archivo CSV
- Visualizaciones interactivas
- Estadísticas descriptivas
- Matriz de correlación
- Histogramas y gráficos de barras

#### 🤖 **Entrenamiento**
- Selecciona el tipo de modelo
- Configura el tamaño del test set
- Entrena el modelo automáticamente
- Visualiza métricas de evaluación
- Ve la importancia de características

#### 🔮 **Predicción**
- Ingresa datos de un estudiante
- Obtén predicción inmediata
- Ve probabilidades de riesgo
- Recibe recomendaciones automáticas

### Uso Programático

```python
from data_preprocessing import prepare_data
from model import DropoutPredictor
import pandas as pd

# Cargar datos
df = pd.read_csv('datos.csv')

# Preparar datos
X_train, X_test, y_train, y_test, features, encoders, scaler = prepare_data(
    df, target_col='estado', test_size=0.2
)

# Crear y entrenar modelo
model = DropoutPredictor(model_type='random_forest')
model.train(X_train, y_train)

# Evaluar
metrics = model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

# Hacer predicción
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Obtener recomendación
recommendation = model.get_recommendation(y_pred_proba[0])
print(recommendation)
```

---

## 📁 Estructura del Proyecto

```
├── app.py                      # Aplicación Streamlit principal
├── model.py                    # Clase DropoutPredictor
├── data_preprocessing.py       # Funciones de preprocesamiento
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Este archivo
```

---

## 📊 Flujo de Trabajo

```
1. CARGA DE DATOS
   ↓
2. EXPLORACIÓN (EDA)
   ↓
3. PREPROCESAMIENTO
   - Manejo de valores faltantes
   - Codificación de categóricas
   - Normalización
   ↓
4. DIVISIÓN TRAIN/TEST
   ↓
5. ENTRENAMIENTO
   - Entrenar múltiples modelos
   - Comparar rendimiento
   ↓
6. EVALUACIÓN
   - Calcular métricas
   - Visualizar resultados
   ↓
7. PREDICCIÓN
   - Hacer predicciones
   - Generar recomendaciones
```

---

## 📈 Métricas de Evaluación

### **Accuracy (Exactitud)**
- Porcentaje de predicciones correctas
- `(TP + TN) / (TP + TN + FP + FN)`

### **Precision (Precisión)**
- De los predichos como abandono, cuántos realmente lo son
- `TP / (TP + FP)`

### **Recall (Sensibilidad)**
- De los que realmente abandonaron, cuántos identificamos
- `TP / (TP + FN)`

### **F1-Score**
- Balance entre Precision y Recall
- `2 * (Precision * Recall) / (Precision + Recall)`

### **ROC-AUC**
- Curva de Características Operativas del Receptor
- Mide la capacidad de discriminación del modelo

---

## 🔗 Fuentes de Datos Recomendadas

- 🌐 **Kaggle**: [Student Dropout Prediction](https://www.kaggle.com)
- 🌐 **UCI Machine Learning Repository**: [Educational Data](https://archive.ics.uci.edu)
- 🌐 **Google Dataset Research**: [Education Analytics](https://datasetsearch.research.google.com)

---

## 📚 Requisitos Previos

- Conceptos básicos de Machine Learning
- Conocimiento de Python
- Familiaridad con Pandas y Scikit-learn

---

## 🎓 Conceptos Clave

- **Clasificación**: Predecir etiquetas discretas
- **Binaria**: Dos clases posibles
- **Supervisado**: Datos etiquetados
- **Overfitting**: Modelo se adapta demasiado a datos de entrenamiento
- **Validación Cruzada**: Técnica para evaluar generalización del modelo
- **Normalización**: Escalar características a rango similar

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si encuentras algún bug o tienes sugerencias:

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para detalles.

---

## 👨‍💻 Autor

**jgarola-dev / Icobo**  
Artificial Intelligence Foundations - Fundació URV 2026

---

## 📧 Contacto

Para preguntas o sugerencias, contacta a través de GitHub.

---

## 🙏 Agradecimientos

- Fundació URV por el curso de AI
- Comunidad de scikit-learn y Streamlit
- Todas las fuentes de datos públicas utilizadas

---

**Última actualización:** Mayo 2026
