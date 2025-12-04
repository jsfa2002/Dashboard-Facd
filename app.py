import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Gasto Nacional en Salud EE. UU. (1960–2023)", layout="wide", initial_sidebar_state="expanded")

# --- TÍTULO Y CONTEXTO PRINCIPAL ---
st.title(" Análisis del Gasto Nacional en Salud de EE. UU. (1960–2023)")
st.markdown("""
<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
<h3> Contexto del Análisis</h3>
<p><strong>Fuente de datos:</strong> Centers for Medicare & Medicaid Services (CMS) - National Health Expenditure Accounts (NHE)</p>
<p><strong>Período analizado:</strong> 1960 - 2023 (64 años de datos históricos)</p>
<p><strong>Unidad de medida:</strong> Millones de dólares estadounidenses (USD)</p>
<p><strong>Objetivo:</strong> Explorar la evolución del gasto en salud de EE. UU. por tipo de servicio y fuente de financiamiento, 
identificando tendencias, patrones y calidad de los datos.</p>
<p>📎 <a href="https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/NationalHealthExpendData/NationalHealthAccountsHistorical" target="_blank">Enlace oficial al dataset</a></p>
</div>
""", unsafe_allow_html=True)

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("nhe2023/NHE2023.csv", encoding="latin1", skiprows=1)
    except UnicodeDecodeError:
        df = pd.read_csv("nhe2023/NHE2023.csv", encoding="utf-8", skiprows=1)
    except FileNotFoundError:
        st.error("No se encontró el archivo NHE2023.csv. Verifica la ruta.")
        st.stop()
    
    df.columns = df.columns.str.strip()
    df.rename(columns={df.columns[0]: "Expenditure_Type"}, inplace=True)
    
    # Transformar de formato ancho a largo
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
    
    df_melt = df_melt.dropna(subset=["Year"])
    return df_melt

nhe = load_data()

# --- SIDEBAR ---
st.sidebar.header(" Configuración del Análisis")
years = st.sidebar.slider(
    " Selecciona rango de años",
    int(nhe["Year"].min()),
    int(nhe["Year"].max()),
    (1980, 2023)
)

show_raw_data = st.sidebar.checkbox(" Mostrar datos crudos", value=False)

filtered = nhe[(nhe["Year"] >= years[0]) & (nhe["Year"] <= years[1])].copy()

# --- MÉTRICAS GENERALES ---
st.header(" Vista General del Dataset")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(" Total de registros", f"{len(nhe):,}")
with col2:
    st.metric(" Categorías únicas", nhe["Expenditure_Type"].nunique())
with col3:
    st.metric(" Años disponibles", f"{int(nhe['Year'].min())} - {int(nhe['Year'].max())}")
with col4:
    missing_pct = (nhe["Amount"].isna().sum() / len(nhe)) * 100
    st.metric(" Datos faltantes", f"{missing_pct:.2f}%")

if show_raw_data:
    st.subheader(" Vista previa de los datos")
    st.dataframe(nhe.head(20), use_container_width=True)

st.markdown("---")

# =========================
# PRIMER RETO
# =========================

st.header(" Primer Reto: Total National Health Expenditures")

st.markdown("""
<div style='background-color: #e8f4f8; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
<h4> Contexto del Ejercicio 1</h4>
<p><strong>Objetivo:</strong> Analizar la evolución del gasto total en salud de Estados Unidos a lo largo del tiempo.</p>
<p><strong>Variable de interés:</strong> Total National Health Expenditures</p>
<p><strong>Actividades:</strong></p>
<ul>
    <li> Diagnóstico de calidad e integridad de datos</li>
    <li> Resúmenes estadísticos descriptivos</li>
    <li> Visualización de tendencias temporales</li>
    <li> Interpretación de resultados y conclusiones</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Filtrar datos
total = filtered[filtered["Expenditure_Type"] == "Total National Health Expenditures"].copy()

if len(total) == 0:
    st.warning(" No hay datos disponibles para el rango de años seleccionado.")
else:
    # 1. Diagnóstico de calidad
    st.subheader(" 1. Diagnóstico de Calidad e Integridad")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(" Registros totales", len(total))
        st.metric(" Valores faltantes", total["Amount"].isna().sum())
    
    with col2:
        st.metric(" Valor mínimo", f"${total['Amount'].min():,.0f}M")
        st.metric(" Valor máximo", f"${total['Amount'].max():,.0f}M")
    
    with col3:
        growth = ((total['Amount'].iloc[-1] - total['Amount'].iloc[0]) / total['Amount'].iloc[0]) * 100
        st.metric(" Crecimiento total", f"{growth:.1f}%")
        avg_annual = ((total['Amount'].iloc[-1] / total['Amount'].iloc[0]) ** (1/len(total)) - 1) * 100
        st.metric(" Crecimiento anual promedio", f"{avg_annual:.2f}%")
    
    # 2. Resumen estadístico
    st.subheader(" 2. Resumen Estadístico")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Estadísticas descriptivas:**")
        stats_df = total["Amount"].describe().to_frame()
        stats_df.columns = ["Monto (Millones USD)"]
        stats_df.index = ["Conteo", "Media", "Desv. Est.", "Mínimo", "Q1 (25%)", "Mediana", "Q3 (75%)", "Máximo"]
        st.dataframe(stats_df.style.format("{:,.2f}"))
    
    with col2:
        # Calcular tasa de crecimiento anual
        total_sorted = total.sort_values("Year")
        total_sorted["Growth_Rate"] = total_sorted["Amount"].pct_change() * 100
        
        st.write("**Últimos 5 años de datos:**")
        recent = total_sorted.tail(5)[["Year", "Amount", "Growth_Rate"]]
        recent.columns = ["Año", "Monto (M USD)", "Crecimiento (%)"]
        st.dataframe(recent.style.format({"Monto (M USD)": "{:,.0f}", "Crecimiento (%)": "{:.2f}"}))
    
    # 3. Visualización
    st.subheader(" 3. Tendencia del Gasto Nacional Total en Salud")
    
    fig_total = go.Figure()
    
    fig_total.add_trace(go.Scatter(
        x=total["Year"],
        y=total["Amount"],
        mode='lines+markers',
        name='Gasto Total',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6),
        hovertemplate='<b>Año:</b> %{x}<br><b>Monto:</b> $%{y:,.0f}M<extra></extra>'
    ))
    
    fig_total.update_layout(
        title="Evolución del Gasto Nacional Total en Salud (Millones USD)",
        xaxis_title="Año",
        yaxis_title="Monto (Millones USD)",
        hovermode='x unified',
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig_total, use_container_width=True)
    
    # Gráfico adicional: Tasa de crecimiento
    fig_growth = go.Figure()
    
    fig_growth.add_trace(go.Bar(
        x=total_sorted["Year"],
        y=total_sorted["Growth_Rate"],
        name='Tasa de Crecimiento',
        marker_color='#2ca02c',
        hovertemplate='<b>Año:</b> %{x}<br><b>Crecimiento:</b> %{y:.2f}%<extra></extra>'
    ))
    
    fig_growth.update_layout(
        title="Tasa de Crecimiento Anual del Gasto (%)",
        xaxis_title="Año",
        yaxis_title="Crecimiento (%)",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig_growth, use_container_width=True)
    
    # 4. Interpretación
    st.subheader(" 4. Interpretación de Resultados")
    
    st.markdown(f"""
    <div style='background-color: #d4edda; padding: 15px; border-radius: 8px; border-left: 5px solid #28a745;'>
    <h5> Hallazgos Principales:</h5>
    <ul>
        <li><strong>Calidad de datos:</strong> El conjunto Total National Health Expenditures presenta <strong>{total["Amount"].isna().sum()} valores faltantes</strong>, 
        indicando excelente integridad de datos.</li>
        <li><strong>Tendencia general:</strong> Se observa una <strong>tendencia creciente clara y sostenida</strong> en el gasto nacional en salud.</li>
        <li><strong>Magnitud del cambio:</strong> El gasto aumentó de <strong>${total['Amount'].iloc[0]:,.0f}M en {int(total['Year'].iloc[0])}</strong> 
        a <strong>${total['Amount'].iloc[-1]:,.0f}M en {int(total['Year'].iloc[-1])}</strong>, representando un incremento del <strong>{growth:.1f}%</strong>.</li>
        <li><strong>Crecimiento promedio:</strong> La tasa de crecimiento anual promedio es del <strong>{avg_annual:.2f}%</strong>, 
        reflejando el aumento constante en costos de atención médica y expansión de cobertura.</li>
        <li><strong>Volatilidad:</strong> La desviación estándar de <strong>${total['Amount'].std():,.0f}M</strong> muestra 
        variabilidad significativa en los montos a lo largo del tiempo.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =========================
# SEGUNDO RETO
# =========================

st.header(" Segundo Reto: Workers' Compensation y Variables Relacionadas")

st.markdown("""
<div style='background-color: #fff3cd; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
<h4> Contexto del Ejercicio 2</h4>
<p><strong>Objetivo:</strong> Analizar el comportamiento del gasto en Workers' Compensation y otras categorías relacionadas 
con seguros y consumo de salud.</p>
<p><strong>Variables de interés:</strong> Workers' Compensation, Health Consumption Expenditures, Net Cost of Health Insurance, entre otras.</p>
<p><strong>Actividades:</strong></p>
<ul>
    <li> Identificación y diagnóstico de datos faltantes</li>
    <li> Análisis comparativo de variables relacionadas</li>
    <li> Resúmenes estadísticos con/sin datos faltantes</li>
    <li> Visualización de evolución temporal múltiple</li>
    <li> Interpretación y conclusiones</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Definir subconjunto
related_vars = ["Workers", "Health", "Insurance", "Consumption"]
sub_nhe = filtered[filtered["Expenditure_Type"].str.contains('|'.join(related_vars), case=False, na=False)].copy()

if len(sub_nhe) == 0:
    st.warning(" No hay datos disponibles para las variables relacionadas en el rango seleccionado.")
else:
    # 1. Diagnóstico de datos faltantes
    st.subheader(" 1. Diagnóstico de Datos Faltantes")
    
    missing_summary = sub_nhe.groupby("Expenditure_Type").agg({
        "Amount": [
            ("Total", "count"),
            ("Faltantes", lambda x: x.isna().sum()),
            ("% Faltantes", lambda x: (x.isna().sum() / len(x)) * 100)
        ]
    }).reset_index()
    
    missing_summary.columns = ["Tipo de Gasto", "Total Registros", "Valores Faltantes", "% Faltantes"]
    missing_summary = missing_summary.sort_values("Valores Faltantes", ascending=False)
    
    st.dataframe(missing_summary.style.format({
        "Total Registros": "{:.0f}",
        "Valores Faltantes": "{:.0f}",
        "% Faltantes": "{:.2f}%"
    }).background_gradient(subset=["% Faltantes"], cmap="Reds"), use_container_width=True)
    
    max_missing = missing_summary.iloc[0]
    st.success(f"**Variable con más valores faltantes:** {max_missing['Tipo de Gasto']} con {int(max_missing['Valores Faltantes'])} valores faltantes ({max_missing['% Faltantes']:.2f}%)")
    
    # Visualización de faltantes
    fig_missing = px.bar(
        missing_summary,
        x="Tipo de Gasto",
        y="% Faltantes",
        title="Porcentaje de Datos Faltantes por Variable",
        color="% Faltantes",
        color_continuous_scale="Reds"
    )
    fig_missing.update_layout(xaxis_tickangle=-45, height=400)
    st.plotly_chart(fig_missing, use_container_width=True)
    
    # 2. Resúmenes básicos
    st.subheader(" 2. Resúmenes Estadísticos")
    
    tab1, tab2 = st.tabs([" Con datos faltantes", " Sin datos faltantes"])
    
    with tab1:
        st.write("**Estadísticas con todos los datos (incluye faltantes):**")
        summary_with = sub_nhe.groupby("Expenditure_Type")["Amount"].describe()
        st.dataframe(summary_with.style.format("{:,.2f}"), use_container_width=True)
    
    with tab2:
        st.write("**Estadísticas excluyendo valores faltantes:**")
        summary_without = sub_nhe.dropna(subset=["Amount"]).groupby("Expenditure_Type")["Amount"].describe()
        st.dataframe(summary_without.style.format("{:,.2f}"), use_container_width=True)
    
    # 3. Visualización comparativa
    st.subheader("📈 3. Evolución Temporal de Variables Relacionadas")
    
    # Selector de variables
    available_vars = sub_nhe["Expenditure_Type"].unique().tolist()
    selected_vars = st.multiselect(
        "Selecciona las variables a comparar:",
        available_vars,
        default=available_vars[:5] if len(available_vars) > 5 else available_vars
    )
    
    if selected_vars:
        sub_filtered = sub_nhe[sub_nhe["Expenditure_Type"].isin(selected_vars)]
        
        fig_related = px.line(
            sub_filtered,
            x="Year",
            y="Amount",
            color="Expenditure_Type",
            title="Comparación del Gasto entre Variables Relacionadas (Millones USD)",
            markers=True,
            line_shape="linear"
        )
        
        fig_related.update_layout(
            xaxis_title="Año",
            yaxis_title="Monto (Millones USD)",
            hovermode='x unified',
            legend_title="Tipo de Gasto",
            template="plotly_white",
            height=600
        )
        
        st.plotly_chart(fig_related, use_container_width=True)
        
        # Gráfico de área apilada
        fig_area = px.area(
            sub_filtered,
            x="Year",
            y="Amount",
            color="Expenditure_Type",
            title="Distribución Acumulada del Gasto"
        )
        fig_area.update_layout(height=500)
        st.plotly_chart(fig_area, use_container_width=True)
    
    # 4. Interpretación
    st.subheader(" 4. Interpretación de Resultados")
    
    st.markdown(f"""
    <div style='background-color: #cce5ff; padding: 15px; border-radius: 8px; border-left: 5px solid #007bff;'>
    <h5> Hallazgos Principales:</h5>
    <ul>
        <li><strong>Calidad de datos:</strong> Se identificaron <strong>{len(missing_summary)} categorías</strong> relacionadas con Workers' Compensation, 
        seguros y consumo de salud.</li>
        <li><strong>Datos faltantes:</strong> La variable con mayor proporción de datos faltantes es <strong>{max_missing['Tipo de Gasto']}</strong> 
        con <strong>{max_missing['% Faltantes']:.2f}%</strong> de valores ausentes.</li>
        <li><strong>Tendencia comparativa:</strong> El gasto en Workers' Compensation se mantiene en niveles <strong>significativamente menores</strong> 
        comparado con gastos en seguros de salud general y consumo médico.</li>
        <li><strong>Consistencia histórica:</strong> Las categorías con mayor cantidad de datos faltantes corresponden generalmente a 
        cambios en metodologías de reporte o categorías implementadas en períodos posteriores.</li>
        <li><strong>Crecimiento diferencial:</strong> Mientras que categorías como Health Consumption muestran crecimiento exponencial, 
        Workers' Compensation presenta una evolución más moderada y estable.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =========================
# CONCLUSIONES GENERALES
# =========================

st.header(" Conclusiones Generales del Análisis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px;'>
    <h5> Fortalezas del Dataset</h5>
    <ul>
        <li>Cobertura temporal extensa (64 años)</li>
        <li>Alta calidad e integridad de datos</li>
        <li>Granularidad detallada por categorías</li>
        <li>Fuente oficial y confiable (CMS)</li>
        <li>Actualización periódica</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background-color: #fff3cd; padding: 15px; border-radius: 8px;'>
    <h5> Limitaciones Identificadas</h5>
    <ul>
        <li>Datos faltantes en categorías específicas</li>
        <li>Cambios metodológicos históricos</li>
        <li>No ajuste por inflación en valores</li>
        <li>Variabilidad en definiciones temporales</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style='background-color: #e7f3ff; padding: 20px; border-radius: 10px; margin-top: 20px;'>
<h4>🔬 Síntesis del Análisis</h4>
<p>El análisis exploratorio de los National Health Expenditures (1960-2023) revela un <strong>crecimiento sostenido y exponencial</strong> 
del gasto en salud de Estados Unidos. Los datos demuestran alta calidad con mínimos valores faltantes en categorías principales.</p>

<p>Se observa una <strong>divergencia significativa</strong> entre categorías: mientras el gasto total y en seguros crece exponencialmente, 
rubros específicos como Workers' Compensation mantienen patrones más estables. Esta diferenciación refleja la naturaleza compleja 
y multifacética del sistema de salud estadounidense.</p>

<p>Las limitaciones identificadas en datos históricos se atribuyen principalmente a cambios metodológicos y actualizaciones 
en la clasificación de categorías de gasto a lo largo del tiempo.</p>
</div>
""", unsafe_allow_html=True)

# =========================
# DESCARGA DE DATOS
# =========================

st.markdown("---")
st.header(" Descarga de Datos Procesados")

col1, col2, col3 = st.columns(3)

with col1:
    csv_total = total.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=" Descargar Total NHE",
        data=csv_total,
        file_name=f'total_nhe_{years[0]}_{years[1]}.csv',
        mime='text/csv'
    )

with col2:
    csv_related = sub_nhe.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=" Descargar Variables Relacionadas",
        data=csv_related,
        file_name=f'related_vars_{years[0]}_{years[1]}.csv',
        mime='text/csv'
    )

with col3:
    csv_full = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=" Descargar Dataset Completo",
        data=csv_full,
        file_name=f'nhe_complete_{years[0]}_{years[1]}.csv',
        mime='text/csv'
    )

# =========================
# FOOTER
# =========================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
<p><strong>Fundamentos para el Análisis de Datos (FACD)</strong></p>
<p>Desarrollado por: <strong>Juan Sebastián Fajardo Acevedo</strong></p>
<p>Docente: <strong>Ana María Gómez Lamus, M.Sc. en Estadística</strong></p>
<p>Universidad de La Sabana - 2025</p>
<p> Datos actualizados al {int(nhe['Year'].max())}</p>
</div>
""", unsafe_allow_html=True)
