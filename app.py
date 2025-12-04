import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Gasto Nacional en Salud EE. UU. (1960–2023)", layout="wide")

st.title("Dashboard: Gasto Nacional en Salud de EE. UU. (1960–2023)")
st.markdown("""
Análisis exploratorio de los National Health Expenditures (NHE) del Centers for Medicare & Medicaid Services (CMS).  
Fuente oficial: [CMS.gov - National Health Expenditure Accounts](https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/NationalHealthExpendData/NationalHealthAccountsHistorical)
""")

# --- Carga de datos ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("nhe2023/NHE2023.csv", encoding="latin1", skiprows=1)
    except UnicodeDecodeError:
        df = pd.read_csv("nhe2023/NHE2023.csv", encoding="utf-8", skiprows=1)
    
    df.columns = df.columns.str.strip()
    df.rename(columns={df.columns[0]: "Expenditure_Type"}, inplace=True)
    
    # Transformar de formato ancho (años como columnas) a largo (columna Year)
    df_melt = df.melt(id_vars=["Expenditure_Type"], var_name="Year", value_name="Amount")
    
    # Limpiar y convertir datos
    df_melt["Year"] = pd.to_numeric(df_melt["Year"], errors="coerce")
    df_melt["Amount"] = (
        df_melt["Amount"]
        .astype(str)
        .str.replace(",", "")
        .str.replace("-", "0")
        .astype(float)
    )
    
    df_melt = df_melt.dropna(subset=["Year", "Amount"])
    return df_melt

nhe = load_data()

# --- Vista general ---
st.subheader("Vista general del dataset")
st.dataframe(nhe.head(10))

# --- Filtros ---
st.sidebar.header("Filtros")
years = st.sidebar.slider(
    "Selecciona rango de años",
    int(nhe["Year"].min()),
    int(nhe["Year"].max()),
    (1980, 2023)
)
filtered = nhe[(nhe["Year"] >= years[0]) & (nhe["Year"] <= years[1])]

# =========================
# PRIMER RETO
# =========================

st.header("Primer Reto: Total National Health Expenditures")

# Definir data frame
total = filtered[filtered["Expenditure_Type"] == "Total National Health Expenditures"]

# Diagnóstico de calidad e integridad
st.subheader("Diagnóstico de calidad e integridad del Total National Health Expenditures")

col1, col2 = st.columns(2)
with col1:
    st.write("Faltantes por columna:")
    st.write(total.isna().sum())

with col2:
    st.write("Resumen estadístico:")
    st.write(total["Amount"].describe())

# Gráfico de tendencia
fig_total = px.line(
    total,
    x="Year",
    y="Amount",
    title="Tendencia del gasto nacional total en salud (Millones USD)",
    markers=True,
    color_discrete_sequence=["#1f77b4"]
)
st.plotly_chart(fig_total, use_container_width=True)

# Interpretación
st.markdown("""
El conjunto Total National Health Expenditures no presenta valores faltantes y muestra una tendencia creciente clara.
La media del gasto nacional se incrementa de forma sostenida a lo largo del tiempo, reflejando el aumento en inversión y costos de salud.
""")

# =========================
# SEGUNDO RETO
# =========================

st.header("Segundo Reto: Workers’ Compensation y variables relacionadas")

# Definir subconjunto de variables de interés
related_vars = ["Workers", "Health", "Insurance", "Consumption"]
sub_nhe = nhe[nhe["Expenditure_Type"].str.contains('|'.join(related_vars), case=False, na=False)]

st.subheader("Subconjunto de variables relacionadas")
st.dataframe(sub_nhe.head(10))

# Diagnóstico de calidad e integridad
st.subheader("Diagnóstico de datos faltantes")
missing_summary = sub_nhe.groupby("Expenditure_Type")["Amount"].apply(lambda x: x.isna().sum()).reset_index()
missing_summary.columns = ["Expenditure_Type", "Missing_Values"]
st.dataframe(missing_summary)

# Variable con más faltantes
max_missing = missing_summary.loc[missing_summary["Missing_Values"].idxmax()]
st.success(f"La variable con más valores faltantes es {max_missing['Expenditure_Type']} ({int(max_missing['Missing_Values'])} valores faltantes).")

# Resúmenes básicos con y sin faltantes
st.subheader("Resúmenes básicos")
st.write("Con datos faltantes:")
st.write(sub_nhe.describe())

st.write("Sin datos faltantes:")
st.write(sub_nhe.dropna().describe())

# Gráfico comparativo
st.subheader("Evolución de variables relacionadas")
fig_related = px.line(
    sub_nhe,
    x="Year",
    y="Amount",
    color="Expenditure_Type",
    title="Comparación del gasto entre variables relacionadas (Workers, Insurance, Health Consumption, etc.)",
    markers=True
)
st.plotly_chart(fig_related, use_container_width=True)

# Interpretación
st.markdown("""
El gasto en Workers’ Compensation mantiene valores menores que los gastos en seguros de salud y consumo médico,
lo cual refleja su papel más específico dentro del sistema de salud.
Las variables con mayor cantidad de datos faltantes corresponden a componentes con cambios históricos en la forma de reporte,
como el Net Cost of Health Insurance Expenditures.
""")

# =========================
# DIAGNÓSTICO GLOBAL
# =========================

st.header("Diagnóstico global del dataset")
col1, col2 = st.columns(2)
with col1:
    st.metric("Filas totales", len(nhe))
    st.metric("Categorías únicas", nhe["Expenditure_Type"].nunique())
with col2:
    st.metric("Año mínimo", int(nhe["Year"].min()))
    st.metric("Año máximo", int(nhe["Year"].max()))

# =========================
# CONCLUSIÓN GENERAL
# =========================

st.header("Conclusión General")
st.markdown("""
El análisis evidencia el crecimiento sostenido del gasto en salud en Estados Unidos desde 1960.
Mientras los costos totales y de seguros aumentan de forma exponencial, otros rubros como Workers’ Compensation se mantienen más estables.
El diagnóstico de datos muestra buena calidad general, aunque existen faltantes en algunas categorías históricas específicas.
""")

st.markdown("---")
st.caption("Desarrollado por: Juan Sebastián Fajardo Acevedo - Fundamentos para el Análisis de Datos (FACD) - 2025")

