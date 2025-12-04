import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Gasto Nacional en Salud EE. UU. (1960–2023)", layout="wide")

st.title(" Dashboard: Gasto Nacional en Salud de EE. UU. (1960–2023)")
st.markdown("""
Análisis exploratorio de los **National Health Expenditures (NHE)** del *Centers for Medicare & Medicaid Services (CMS)*.  
Fuente: [CMS.gov - National Health Expenditure Accounts](https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/NationalHealthExpendData/NationalHealthAccountsHistorical)
""")

# --- Carga de datos robusta ---
@st.cache_data
def load_data():
    try:
        # Saltamos la primera fila del título y usamos la segunda como encabezado
        df = pd.read_csv("nhe2023/NHE2023.csv", encoding="latin1", skiprows=1)
    except UnicodeDecodeError:
        df = pd.read_csv("nhe2023/NHE2023.csv", encoding="utf-8", skiprows=1)
    
    # Limpiar nombres de columnas
    df.columns = df.columns.str.strip()
    
    # Renombrar la primera columna
    df.rename(columns={df.columns[0]: "Expenditure_Type"}, inplace=True)
    
    # Transformar de formato ancho (años como columnas) a largo (columna Year)
    df_melt = df.melt(id_vars=["Expenditure_Type"], var_name="Year", value_name="Amount")
    
    # Convertir tipos
    df_melt["Year"] = pd.to_numeric(df_melt["Year"], errors="coerce")
    df_melt["Amount"] = (
        df_melt["Amount"]
        .astype(str)
        .str.replace(",", "")
        .str.replace("-", "0")
        .astype(float)
    )
    
    # Eliminar filas sin año o sin monto
    df_melt = df_melt.dropna(subset=["Year", "Amount"])
    
    return df_melt

nhe = load_data()

# Vista previa
st.subheader(" Vista general del dataset")
st.dataframe(nhe.head(10))

# --- Filtros ---
st.sidebar.header(" Filtros")
years = st.sidebar.slider(
    "Selecciona rango de años",
    int(nhe["Year"].min()),
    int(nhe["Year"].max()),
    (1980, 2023)
)

filtered = nhe[(nhe["Year"] >= years[0]) & (nhe["Year"] <= years[1])]

# --- Tendencia total ---
st.header("1️ Evolución del gasto nacional total")

total = filtered[filtered["Expenditure_Type"] == "Total National Health Expenditures"]

fig_total = px.line(
    total,
    x="Year",
    y="Amount",
    title="Tendencia del gasto nacional total en salud (Millones USD)",
    markers=True,
    color_discrete_sequence=["#1f77b4"]
)
st.plotly_chart(fig_total, use_container_width=True)

# --- Workers’ Compensation ---
st.header("2️ Gasto en Workers’ Compensation")

workers = filtered[filtered["Expenditure_Type"].str.contains("Workers", case=False, na=False)]

fig_workers = px.line(
    workers,
    x="Year",
    y="Amount",
    title="Tendencia del gasto en Workers’ Compensation",
    markers=True,
    color_discrete_sequence=["#e45756"]
)
st.plotly_chart(fig_workers, use_container_width=True)

# --- Diagnóstico y resumen ---
st.header("3️ Diagnóstico de datos")
col1, col2 = st.columns(2)
with col1:
    st.metric("Filas totales", len(nhe))
    st.metric("Categorías únicas", nhe["Expenditure_Type"].nunique())
with col2:
    st.metric("Año mínimo", int(nhe["Year"].min()))
    st.metric("Año máximo", int(nhe["Year"].max()))

# --- Conclusiones ---
st.header("4️ Conclusiones e Insights")
st.markdown("""
-  El gasto nacional total muestra una **tendencia creciente sostenida** desde 1960, superando los 4.8 billones USD en 2023.  
-  El gasto en **Workers’ Compensation** mantiene una tendencia estable, con variaciones moderadas.  
-  Este dataset ilustra cómo el gasto público y privado en salud ha evolucionado a lo largo de seis décadas.  
-  Los datos pueden ampliarse a otros componentes (Medicare, Medicaid, etc.) para análisis detallados.
""")

st.markdown("---")
st.caption("Desarrollado por [Tu Nombre] — Fundamentos para el Análisis de Datos (FACD) · 2025")
