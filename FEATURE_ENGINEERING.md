````markdown
# 🔧 Feature Engineering - Guía Completa

## 📚 Introducción

El Feature Engineering es el proceso de seleccionar, transformar y crear características (features) de los datos crudos para mejorar el rendimiento de los modelos de Machine Learning. Este documento proporciona una guía completa sobre las técnicas y análisis implementadas en nuestro proyecto.

---

## 📊 1. Análisis Univariado (Una Variable)

### 1.1 Variables Numéricas

#### Medidas de Tendencia Central
```
Media (μ) = Σx / n
Mediana = Valor central cuando se ordenan los datos
Moda = Valor más frecuente
```

#### Medidas de Dispersión
```
Desviación Estándar (σ) = √[Σ(x - μ)² / n]
Varianza (σ²) = [Σ(x - μ)² / n]
Rango Intercuartílico (IQR) = Q3 - Q1
```

#### Medidas de Forma
```
Asimetría (Skewness):
- ≈ 0: Distribución simétrica
- > 0: Asimetría a la derecha
- < 0: Asimetría a la izquierda

Curtosis (Kurtosis):
- ≈ 0: Normal (mesokúrtica)
- > 0: Colas pesadas (leptokúrtica)
- < 0: Colas ligeras (platikúrtica)
```

### 1.2 Variables Categóricas

#### Análisis de Frecuencias
```
Frecuencia Absoluta = Número de ocurrencias
Frecuencia Relativa (%) = (Frecuencia / Total) × 100
```

#### Detección de Desequilibrio
```
Ratio Desequilibrio = Clase_Mayoritaria / Clase_Minoritaria
- < 1.5: Balanceado
- 1.5 - 3: Moderadamente desequilibrado
- > 3: Altamente desequilibrado
```

---

## 📈 2. Análisis Bivariado (Dos Variables)

### 2.1 Correlación de Pearson

**Fórmula:**
```
r = Σ[(x - x̄)(y - ȳ)] / √[Σ(x - x̄)² × Σ(y - ȳ)²]
```

**Interpretación:**
```
r ∈ [-1, 1]
- r = 1: Correlación positiva perfecta
- r > 0.7: Correlación positiva fuerte
- 0.3 < r < 0.7: Correlación positiva moderada
- r ≈ 0: Sin correlación
- -0.3 > r > -0.7: Correlación negativa moderada
- r < -0.7: Correlación negativa fuerte
- r = -1: Correlación negativa perfecta
```

### 2.2 Diagrama de Dispersión (Scatter Plot)

Visualización de relaciones entre dos variables numéricas:
- **OLS (Ordinary Least Squares)**: Línea de tendencia
- Identifica clusters y outliers
- Detecta relaciones no lineales

---

## 🎯 3. Análisis Multivariado (Múltiples Variables)

### 3.1 Matriz de Correlación

Tabla de correlaciones entre todos los pares de variables:
```
        Var1   Var2   Var3
Var1     1.0   0.45  -0.12
Var2    0.45    1.0   0.78
Var3   -0.12   0.78    1.0
```

### 3.2 Heatmap

Visualización con código de colores:
- 🔵 **Azul**: Correlación negativa
- ⚪ **Blanco**: Sin correlación
- 🔴 **Rojo**: Correlación positiva

### 3.3 Matriz de Dispersión (Scatter Matrix)

- Todos los pares de variables
- Histogramas en la diagonal
- Scatter plots fuera de la diagonal
- Útil para identificar multicolinealidad

---

## 🚨 4. Detección de Problemas

### 4.1 Outliers (Valores Atípicos)

#### Método IQR (Interquartile Range)
```
Q1 = Percentil 25
Q3 = Percentil 75
IQR = Q3 - Q1

Límite Inferior = Q1 - 1.5 × IQR
Límite Superior = Q3 + 1.5 × IQR

Outlier = Valor < Límite Inferior O Valor > Límite Superior
```

### 4.2 Multicolinealidad

Cuando dos o más variables están altamente correlacionadas:
```
Problema: r > 0.9 (muy alta correlación)
Impacto: Inestabilidad en coeficientes del modelo
Solución: Eliminar una de las variables
```

### 4.3 Valores Faltantes

```
Patrones Comunes:
- Missing Completely At Random (MCAR): Aleatorio puro
- Missing At Random (MAR): Depende de otras variables
- Not Missing At Random (NMAR): Depende de la variable misma
```

---

## ⚖️ 5. Análisis de Desequilibrio de Clases

### 5.1 Identificación

```
Distribución de Clases:
- Dropout: 1200 (40%)
- Enrolled: 1500 (50%)
- Graduate: 300 (10%)

Ratio Desequilibrio = 1500 / 300 = 5:1
Estado: DESEQUILIBRADO (5:1)
```

### 5.2 Estrategias de Manejo

#### A) Resampling
```
SMOTE (Synthetic Minority Oversampling):
- Genera sintéticamente nuevos ejemplos de la clase minoritaria
- Mejora el balance sin perder información

Random Oversampling:
- Replica aleatoriamente ejemplos de la clase minoritaria
- Simple pero puede causar overfitting

Random Undersampling:
- Reduce ejemplos de la clase mayoritaria
- Pierde información importante
```

#### B) Weighted Methods
```
Class Weight = 1 / Frecuencia_Relativa

Ejemplo:
- Dropout (40%): weight = 1 / 0.4 = 2.5
- Enrolled (50%): weight = 1 / 0.5 = 2.0
- Graduate (10%): weight = 1 / 0.1 = 10.0
```

#### C) Métricas Apropiadas
```
❌ NO usar: Accuracy (engañosa con clases desequilibradas)
✅ Usar: 
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: 2 × (Precision × Recall) / (Precision + Recall)
- ROC-AUC: Area bajo la curva ROC
```

---

## 📊 6. Interpretación de Gráficos

### 6.1 Histograma
```
Muestra: Distribución de frecuencias
Útil para: Identificar sesgo y multimodalidad
Interpretación:
- Un pico: Distribución unimodal
- Múltiples picos: Multimodal (datos de subgrupos)
- Sesgo izquierdo: Cola izquierda larga
- Sesgo derecho: Cola derecha larga
```

### 6.2 Box Plot
```
Muestra: Resumen de cinco números
Componentes:
- Caja: Q1 a Q3 (50% central)
- Línea media: Mediana
- Bigotes: Min a Max (sin outliers)
- Puntos: Outliers
```

### 6.3 Scatter Plot
```
Muestra: Relación entre dos variables
Con OLS: Línea de tendencia
Interpretación:
- Nube ascendente: Correlación positiva
- Nube descendente: Correlación negativa
- Nube dispersa: Sin correlación
- Agrupamientos: Clusters o subgrupos
```

---

## 🔢 7. Fórmulas Clave

### 7.1 Estadísticas Descriptivas

```
Media: μ = Σx / n

Mediana: M = {
    x[(n+1)/2] si n es impar
    [x[n/2] + x[n/2+1]] / 2 si n es par
}

Varianza: σ² = Σ(x - μ)² / n

Desv. Est.: σ = √σ²

Coef. Variación: CV = σ / μ × 100%
```

### 7.2 Correlación

```
Pearson: r = Cov(X,Y) / (σx × σy)

Spearman: ρ = Correlación Pearson de rangos
```

### 7.3 Métricas de Clasificación

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

---

## 💡 8. Buenas Prácticas

### 8.1 Exploración de Datos
✅ Siempre comenzar con análisis exploratorio
✅ Visualizar antes de modelar
✅ Documentar hallazgos
✅ Identificar problemas tempranamente

### 8.2 Tratamiento de Datos
✅ Manejar valores faltantes estratégicamente
✅ Detectar y tratar outliers
✅ Considerar transformaciones (log, sqrt)
✅ Normalizar/Estandarizar si es necesario

### 8.3 Feature Engineering
✅ Eliminar features correlacionadas
✅ Crear features significativas
✅ Considerar el conocimiento del dominio
✅ Validar importancia de features

### 8.4 Manejo de Desequilibrio
✅ Detectar desequilibrio de clases
✅ Elegir estrategia apropiada
✅ Usar métricas adecuadas
✅ Validar con stratified cross-validation

---

## 🎯 9. Checklist de Feature Engineering

```
□ Descripción general del dataset
□ Análisis de valores faltantes
□ Análisis de tipos de datos
□ Estadísticas descriptivas
□ Visualización univariada
□ Análisis de correlaciones
□ Matriz de dispersión
□ Detección de outliers
□ Análisis de desequilibrio
□ Feature importance
□ Transformaciones necesarias
□ Validación de supuestos
□ Documentación de hallazgos
```

---

## 📚 10. Referencias y Recursos

### Conceptos Básicos
- Machine Learning: https://scikit-learn.org/
- Pandas: https://pandas.pydata.org/
- Plotly: https://plotly.com/python/

### Feature Engineering
- Kaggle Feature Engineering: https://www.kaggle.com/learn/feature-engineering
- Imbalanced Learning: https://imbalanced-learn.org/

### Estadística
- Teoría de correlaciones
- Distribuciones de probabilidad
- Pruebas de hipótesis

---

**Última actualización:** 2026-05-08
**Autor:** ML Team - Artificial Intelligence Foundations
**Institución:** Fundació URV
````
