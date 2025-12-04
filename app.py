import streamlit as st
import pandas as pd
import plotly.express as px

# Configuración de página
st.set_page_config(page_title="Gasto Nacional en Salud EE. UU. (1960–2023)", layout="wide")

st.title(" Dashboard: Gasto Nacional en Salud de EE. UU. (1960–2023)")
st.markdown("""
Análisis exploratorio y diagnóstico de datos basado en los **National Health Expenditures (NHE)** del *Centers for Medicare & Medicaid Services (CMS)*.  
Fuente oficial: [CMS.gov - National Health Expenditure Accounts](https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/NationalHealthExpendData/NationalHealthAccountsHistorical)
""")

# --- Cargar datos ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/NHE_TypeOfService_SourceOfFunds_1960_2023.csv")
    return df

nhe = load_data()

st.sidebar.header("Filtros")
years = st.sidebar.slider("Selecciona rango de años", int(nhe["Year"].min()), int(nhe["Year"].max()), (1980, 2023))
nhe = nhe[(nhe["Year"] >= years[0]) & (nhe["Year"] <= years[1])]

# --- Diagnóstico de calidad ---
st.header("1️ Diagnóstico de calidad e integridad de datos")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Vista general del dataset")
    st.dataframe(nhe.head())
with col2:
    st.subheader("Estadísticas básicas")
    st.write(nhe.describe())

# Faltantes
st.subheader("Valores faltantes por variable")
missing = nhe.isna().mean().sort_values(ascending=False)
st.bar_chart(missing)

# --- Total National Health Expenditures ---
st.header("2️ Evolución del Gasto Nacional Total")

if "Total National Health Expenditures" in nhe.columns:
    fig = px.line(
        nhe,
        x="Year",
        y="Total National Health Expenditures",
        title="Evolución del gasto nacional en salud (Millones USD)",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("⚠️ No se encontró la columna 'Total National Health Expenditures' en el dataset.")

# --- Workers’ Compensation ---
st.header("3️ Gasto en Workers’ Compensation")
workers_cols = [c for c in nhe.columns if "Workers" in c]
if workers_cols:
    df_workers = nhe[["Year"] + workers_cols]
    fig2 = px.line(
        df_workers,
        x="Year",
        y=workers_cols,
        title="Tendencia de gasto en Workers’ Compensation por tipo",
        markers=True
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning(" No se encontraron variables relacionadas con 'Workers’ Compensation'.")

# --- Conclusiones ---
st.header("4️ Conclusiones e Insights")
st.markdown("""
-  El gasto nacional total muestra una **tendencia creciente sostenida** desde 1960.  
-  Las compensaciones laborales (Workers’ Compensation) representan una **fracción pequeña pero estable** del gasto total.  
-  Se detectan **faltantes históricos** en algunas variables, reflejando cambios en la cobertura o en las políticas de registro.  
-  Este ejercicio ejemplifica cómo transformar datos en información útil para **política pública y análisis económico**.
""")

st.markdown("---")
st.caption("Desarrollado por [Tu Nombre] — Fundamentos para el Análisis de Datos (FACD) · 2025")
