# Dashboard de Análisis de Gasto Nacional en Salud (NHE 1960-2023)

Este repositorio contiene la preparación, limpieza y análisis de datos del National Health Expenditure (NHE). El proyecto fue desarrollado como una aplicación web interactiva utilizando Streamlit, permitiendo no solo la visualización histórica sino también la proyección de tendencias futuras mediante modelos de Machine Learning y técnicas de aprendizaje no supervisado.

---

## Contexto del Proyecto

Asignatura: Fundamentos para el Análisis de Datos (FACD)
Institución: Universidad Escuela Colombiana de Ingeniería Julio Garavito
Fuente de Datos: Centers for Medicare & Medicaid Services (CMS) - [NHE2023]

El objetivo principal del taller fue aplicar técnicas para evaluar, limpiar y transformar datos crudos en información útil, siguiendo el ciclo de vida de la ciencia de datos. Se trabajó con el dataset histórico de gastos en salud de EE. UU. (1960-2023), enfocándose en la detección de anomalías, tratamiento de valores nulos, segmentación de categorías y análisis de tendencias.

---

## Objetivos

El desarrollo se centró en resolver dos retos principales planteados en el proyecto:

### 1. Primer Reto: Análisis del Gasto Total (Total National Health Expenditures)
El objetivo fue definir un dataframe específico para esta variable y realizar:
* Diagnóstico de Calidad: Evaluar integridad, completitud y consistencia estadística del subconjunto.
* Resúmenes Estadísticos: Cálculo de métricas clave incluyendo tasas de crecimiento compuesto (CAGR).
* Análisis de Robustez: Detección de valores atípicos (outliers) en las tasas de crecimiento.
* Visualización e Interpretación: Representación gráfica de la tendencia temporal y análisis del comportamiento exponencial del gasto.

### 2. Segundo Reto: Análisis Comparativo (Workers' Compensation y Variables Relacionadas)
El objetivo fue explorar variables específicas dentro del rango de "Health Consumption Expenditures" a "Net Cost of Health Insurance", con énfasis en Workers' Compensation:
* Diagnóstico de Integridad: Evaluación de la continuidad de las series temporales seleccionadas.
* Segmentación de Categorías: Agrupamiento de variables según su comportamiento financiero.
* Correlación Estructural: Análisis de relaciones entre variables eliminando el efecto de tendencia temporal.
* Representación Gráfica: Visualización comparativa multidimensional.

---

## Implementación Técnica

La solución fue construida en Python utilizando un enfoque modular. A continuación se describen las características clave de la implementación:

### Procesamiento de Datos (ETL)
* Estrategia de Datos Faltantes: Se implementó una política de limpieza donde se eliminaron aquellos registros que carecían simultáneamente de información temporal (Año) y valor financiero (Monto). Esta decisión se tomó debido a que la ausencia de estas variables críticas impide la realización de cualquier análisis de series de tiempo o cuantificación económica válida.
* Normalización: Estandarización de nombres de columnas y limpieza de caracteres especiales en montos ($, , -).
* Transformación: Conversión de formato ancho a largo (melt) para facilitar el procesamiento de series temporales.
* Control de duplicidad: Eliminación de registros redundantes basados en la llave compuesta de tipo de gasto y año.

### Análisis Exploratorio y Visualización Avanzada
La aplicación incorpora técnicas estadísticas avanzadas:
* Detección de Outliers (IQR): Implementación del método de Rango Intercuartílico sobre las tasas de crecimiento anual para identificar anomalías estadísticas.
* Clustering (K-Means): Algoritmo de aprendizaje no supervisado para segmentar las categorías de gasto en grupos funcionales basándose en su volumen y CAGR.
* Filtrado Dinámico: Selección interactiva de variables para análisis comparativo.

### Modelado y Proyecciones (Forecasting)
Para agregar valor al análisis histórico, se implementó un módulo de predicción que proyecta el gasto a futuro utilizando una estrategia de Ensemble que combina tres modelos matemáticos:
1. Suavizado Exponencial Triple (Holt-Winters): Para capturar nivel y tendencia reciente.
2. Regresión Polinomial (Grado 3): Para capturar la convexidad y aceleración no lineal de la curva.
3. Regresión Lineal: Para establecer una línea base conservadora.

El modelo final ponderado incluye intervalos de confianza del ±15% para cuantificar la incertidumbre futura.

### Interfaz de Usuario
* Personalización Temática: Diseño adaptado con una paleta de colores institucional (USA Theme) para mejorar la legibilidad y el contexto geográfico del análisis.

---

## Link en línea

* https://dashboard-facd-eeuu-mahvjsfa.streamlit.app/

---

## Instrucciones de Ejecución

1. Clonar el repositorio:
   git clone https://github.com/jsfa2002/Dashboard-Facd.git
   cd DASHBOARD-FACD

2. Instalar dependencias:
   Se recomienda usar un entorno virtual.
   pip install -r requirements.txt

3. Ejecutar la aplicación:
   python -m streamlit run app.py

---

## Autores

* Juan Sebastián Fajardo Acevedo
* Miguel Ángel Hernández Vargas