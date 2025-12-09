# Dashboard de Análisis de Gasto Nacional en Salud (NHE 1960-2023)

Este repositorio contiene la preparación, limpieza y análisis de datos del **National Health Expenditure (NHE)**. El proyecto fue desarrollado como una aplicación web interactiva utilizando **Streamlit**, permitiendo no solo la visualización histórica sino también la proyección de tendencias futuras mediante modelos de Machine Learning.

---

## Contexto del Proyecto

**Asignatura:** Fundamentos para el Análisis de Datos (FACD)  
**Institución:** Universidad Escuela Colombiana de Ingeniería Julio Garavito  
**Fuente de Datos:** Centers for Medicare & Medicaid Services (CMS) - [NHE2023]

El objetivo principal del taller fue aplicar técnicas de **Data Quality** y **Data Wrangling** para transformar datos crudos en información de valor, siguiendo el ciclo de vida de la ciencia de datos. Se trabajó con el dataset histórico de gastos en salud de EE. UU. (1960-2023), enfocándose en la detección de anomalías, tratamiento de valores nulos y análisis de tendencias[cite: 485].

---

## Objetivos

El desarrollo se centró en resolver dos retos principales planteados en la guía de laboratorio:

### 1. Primer Reto: Análisis del Gasto Total (Total National Health Expenditures)
El objetivo fue definir un dataframe específico para esta variable y realizar:
* **Diagnóstico de Calidad:** Evaluar integridad y completitud del subconjunto[cite: 487].
* **Resúmenes Estadísticos:** Cálculo de métricas clave (Media, Desviación, CAGR)[cite: 487].
* **Visualización:** Representación gráfica de la tendencia temporal (1960-2023)[cite: 488].
* **Interpretación:** Análisis del crecimiento sostenido del gasto[cite: 489].

### 2. Segundo Reto: Análisis Comparativo (Workers' Compensation y Variables Relacionadas)
El objetivo fue explorar variables específicas dentro del rango de "Health Consumption Expenditures" a "Net Cost of Health Insurance", con énfasis en **Workers' Compensation**:
* **Diagnóstico de Datos Faltantes:** Identificación de variables con mayor cantidad de vacíos[cite: 491, 492].
* **Análisis con/sin Imputación:** Comparación de resúmenes estadísticos incluyendo y excluyendo valores nulos[cite: 493].
* **Representación Gráfica:** Visualización comparativa de múltiples categorías relacionadas con seguros y consumo[cite: 494].

---

## Implementación Técnica

La solución fue construida en **Python** utilizando un enfoque modular. A continuación se describen las características clave de la implementación:

### Procesamiento de Datos (ETL)
* **Limpieza:**
    * Normalización de nombres de columnas y eliminación de caracteres especiales en montos ($, , -).
    * Transformación de formato *ancho* a *largo* (`melt`) para facilitar el análisis temporal.
    * Filtrado de filas irrelevantes.
    * Eliminación de registros duplicados según el tipo de gasto y el año.

### Análisis Exploratorio y Visualización
* **Filtrado Dinámico:** Selección de variables relacionadas mediante palabras clave (`Workers`, `Health`, `Insurance`, `Consumption`) para el análisis comparativo.
* **Visualizaciones Interactivas:**
    * Gráficos de líneas con *hover* detallado.
    * Gráficos de área apilada para analizar la composición del gasto.
    * Heatmaps para visualizar la densidad de datos faltantes por categoría.
    * Selectores interactivos en el sidebar para rangos de años (1960-2023).

### Modelado y Proyecciones
Para agregar valor al análisis histórico, se implementó un módulo de predicción que proyecta el gasto a futuro (5-20 años) utilizando una estrategia de **Ensemble** que combina:
1.  **Suavizado Exponencial Triple (Holt-Winters):** Para capturar nivel y tendencia.
2.  **Regresión Polinomial (Grado 3):** Para capturar curvaturas no lineales.
3.  **Regresión Lineal:** Para moderar tendencias explosivas.

El modelo calcula intervalos de confianza del ±15% para las proyecciones futuras.

---

## Instrucciones de Ejecución

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/jsfa2002/Dashboard-Facd.git
    cd DASHBOARD-FACD
    ```

2.  **Instalar dependencias:**
    Se recomienda usar un entorno virtual.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Ejecutar la aplicación:**
    ```bash
    python -m streamlit run app.py
    ```

---

## Autores

* **Juan Sebastián Fajardo Acevedo**
* **Miguel Ángel Hernández Vargas**
