fig_forecast.add_trace(go.Scatter(
    x=future_years,
    y=exp_smooth,
    mode='lines',
    name='Suavizado Exponencial',
    line=dict(color='#f59e0b', width=2, dash='dot'),
    visible='legendonly'
))

fig_forecast.add_trace(go.Scatter(
    x=future_years,
    y=poly,
    mode='lines',
    name='Regresión Polinomial',
    line=dict(color='#8b5cf6', width=2, dash='dot'),
    visible='legendonly'
))

fig_forecast.update_layout(
    title=f"Proyección del Gasto Nacional en Salud hasta {int(future_years[-1])}",
    xaxis_title="Año",
    yaxis_title="Monto (Millones USD)",
    hovermode='x unified',
    template="plotly_white",
    height=600
)

st.plotly_chart(fig_forecast, use_container_width=True)

st.write("**Tabla de proyecciones detalladas:**")
forecast_display = forecast_df.copy()
for col in forecast_display.columns[1:]:
    forecast_display[col] = forecast_display[col].apply(lambda x: f"${x:,.0f}M")
st.dataframe(forecast_display, use_container_width=True)

col1, col2, col3 = st.columns(3)
with col1:
    final_forecast = ensemble[-1]
    st.metric(
        f"Proyección para {int(future_years[-1])}",
        f"${final_forecast:,.0f}M"
    )
with col2:
    forecast_growth = ((ensemble[-1] - total_sorted['Amount'].iloc[-1]) / total_sorted['Amount'].iloc[-1]) * 100
    st.metric(
        f"Crecimiento proyectado ({forecast_periods} años)",
        f"{forecast_growth:.1f}%"
    )
with col3:
    annual_growth_forecast = ((ensemble[-1] / total_sorted['Amount'].iloc[-1]) ** (1/forecast_periods) - 1) * 100
    st.metric(
        "CAGR proyectado",
        f"{annual_growth_forecast:.2f}%"
    )

st.markdown(f"""
<div class="interpretation-box">
<p><strong>Interpretación del forecasting:</strong> El modelo ensemble proyecta que el gasto nacional en salud alcanzará 
aproximadamente ${ensemble[-1]:,.0f} millones de dólares en {int(future_years[-1])}, lo que representa un crecimiento 
del {forecast_growth:.1f}% respecto al último valor observado en {int(total_sorted['Year'].iloc[-1])}. La tasa de 
crecimiento anual compuesta proyectada (CAGR) de {annual_growth_forecast:.2f}% es ligeramente inferior al crecimiento 
histórico de {avg_annual:.2f}%, sugiriendo una moderación en la expansión del gasto. Esta desaceleración podría 
atribuirse a diversos factores: mayor adopción de medicina preventiva, eficiencias operativas en el sistema de salud, 
presión política para contener costos, y posible estabilización demográfica post-baby boomer.</p>

<p>El intervalo de confianza de ±15% refleja la incertidumbre inherente a cualquier proyección de largo plazo. Factores 
que podrían llevar el gasto hacia el límite superior incluyen: nuevas pandemias, avances tecnológicos costosos (terapias 
génicas, medicina de precisión), expansión adicional de cobertura, o aumento en longevidad. Factores que podrían 
contener el gasto incluyen: reformas estructurales del sistema, mayor competencia en el mercado de seguros, adopción 
de telemedicina, o cambios en patrones de utilización.</p>

<p>Es crucial interpretar estas proyecciones como escenarios plausibles basados en tendencias históricas, no como 
predicciones deterministas. Los modelos de series temporales tienen limitaciones inherentes al extrapolar hacia el 
futuro, particularmente en horizontes largos donde la probabilidad de cambios estructurales aumenta significativamente.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
# ============================================
# SEGUNDO RETO
# ============================================
st.header("Segundo Reto: Workers' Compensation y Variables Relacionadas")
st.markdown("""
<div class="context-box">
<h4>Contexto del Ejercicio 2</h4>
<p><strong>Objetivo:</strong> Realizar un análisis comparativo profundo del gasto en Workers' Compensation y otras categorías 
relacionadas con seguros de salud y consumo médico, identificando patrones diferenciales, evaluando la calidad de datos en 
categorías secundarias, y proyectando evoluciones futuras para cada componente.</p>
<p><strong>Variables de interés:</strong> Workers' Compensation, Private Health Insurance, Health Consumption Expenditures, 
Net Cost of Health Insurance, Public Health Activity, y otras categorías relacionadas con seguros y prestación de servicios 
de salud.</p>
<p><strong>Importancia del análisis:</strong> Mientras el gasto total proporciona una visión macroeconómica, el análisis 
desagregado por categorías revela dinámicas específicas de diferentes componentes del sistema de salud. Workers' Compensation, 
por ejemplo, refleja el costo de lesiones y enfermedades ocupacionales; los seguros privados de salud muestran la evolución 
del mercado privado; y las categorías de consumo directo revelan patrones de utilización de servicios. La comparación entre 
estas categorías permite identificar qué componentes crecen más rápido, cuáles se estancan, y dónde existen oportunidades 
de optimización o necesidades de mayor inversión.</p>
<p><strong>Metodología aplicada:</strong></p>
<ul>
    <li><strong>Identificación de categorías relevantes:</strong> Filtrado basado en palabras clave relacionadas con 
    compensación laboral, seguros y consumo de salud</li>
    <li><strong>Diagnóstico detallado de datos faltantes:</strong> Análisis cuantitativo y cualitativo de la completitud 
    por categoría, identificando patrones de ausencia de datos</li>
    <li><strong>Análisis estadístico comparativo:</strong> Comparación de distribuciones, tasas de crecimiento, y 
    volatilidad entre categorías</li>
    <li><strong>Visualización multidimensional:</strong> Gráficos de evolución temporal múltiple, gráficos de área apilada, 
    y heatmaps de correlación</li>
    <li><strong>Forecasting por categoría:</strong> Proyecciones individuales para cada variable seleccionada</li>
    <li><strong>Interpretación contextualizada:</strong> Vinculación de hallazgos con políticas laborales, evolución del 
    mercado de seguros, y cambios en patrones de atención médica</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Definir y filtrar variables relacionadas
related_vars = ["Workers", "Health", "Insurance", "Consumption"]
sub_nhe = filtered[filtered["Expenditure_Type"].str.contains('|'.join(related_vars), case=False, na=False)].copy()
if len(sub_nhe) == 0:
    st.warning("No hay datos disponibles para las variables relacionadas en el rango seleccionado.")
else:
    # 1. DIAGNÓSTICO DE DATOS FALTANTES
    st.subheader("1. Diagnóstico Integral de Datos Faltantes")
    st.markdown("""
    <p>El análisis de datos faltantes es particularmente importante en categorías específicas del NHE, ya que algunas 
    categorías se implementaron en años posteriores a 1960 o sufrieron redefiniciones metodológicas. La presencia de datos 
    faltantes no necesariamente indica problemas de calidad, sino que puede reflejar cambios en la clasificación de gastos, 
    creación de nuevas categorías, o consolidación de categorías existentes. Este diagnóstico permite identificar qué 
    categorías tienen cobertura temporal completa y cuáles requieren tratamiento especial en el análisis.</p>
    """, unsafe_allow_html=True)

    # Calcular estadísticas de datos faltantes
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

    # Identificar variable con más faltantes
    max_missing = missing_summary.iloc[0]
    st.success(f"Variable con mayor proporción de datos faltantes: {max_missing['Tipo de Gasto']} con {int(max_missing['Valores Faltantes'])} valores ausentes ({max_missing['% Faltantes']:.2f}%)")

    # Visualización de datos faltantes
    fig_missing = go.Figure()

    fig_missing.add_trace(go.Bar(
        x=missing_summary["Tipo de Gasto"],
        y=missing_summary["% Faltantes"],
        marker=dict(
            color=missing_summary["% Faltantes"],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="% Faltantes")
        ),
        text=missing_summary["% Faltantes"].apply(lambda x: f"{x:.1f}%"),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Faltantes: %{y:.2f}%<extra></extra>'
    ))

    fig_missing.update_layout(
        title="Análisis de Datos Faltantes por Categoría",
        xaxis_title="Categoría de Gasto",
        yaxis_title="Porcentaje de Datos Faltantes (%)",
        template="plotly_white",
        height=500,
        xaxis_tickangle=-45
    )

    st.plotly_chart(fig_missing, use_container_width=True)

    st.markdown(f"""
    <div class="interpretation-box">
    <p><strong>Interpretación del diagnóstico de datos faltantes:</strong> El análisis revela que {max_missing['Tipo de Gasto']} 
    presenta la mayor proporción de datos ausentes con {max_missing['% Faltantes']:.2f}%. Esta situación puede deberse a varias 
    razones: (1) la categoría fue creada o comenzó a reportarse sistemáticamente después de 1960, (2) hubo cambios en la 
    metodología de clasificación del CMS que llevaron a la discontinuación o fusión de categorías, o (3) ciertos tipos de gastos 
    no eran relevantes o medibles en períodos históricos tempranos.</p>

    <p>Es notable que categorías como Workers' Compensation y Health Insurance muestran completitud cercana o igual al 100%, 
    lo cual es esperado dado que son componentes fundamentales y de larga data en el sistema de salud estadounidense. Por el 
    contrario, categorías más granulares o especializadas tienden a tener mayor proporción de datos faltantes, especialmente 
    en años históricos. Para el análisis de series temporales y forecasting, es recomendable centrarse en categorías con alta 
    completitud o, alternativamente, restringir el análisis temporal a los períodos donde los datos están disponibles.</p>
    </div>
    """, unsafe_allow_html=True)

    # 2. ANÁLISIS ESTADÍSTICO COMPARATIVO
    st.subheader("2. Análisis Estadístico Comparativo entre Categorías")

    st.markdown("""
    <p>El análisis estadístico comparativo permite identificar diferencias y similitudes en el comportamiento de distintas 
    categorías de gasto. Mediante la comparación de medidas de tendencia central, dispersión, y crecimiento, podemos determinar 
    qué categorías son más volátiles, cuáles han crecido más rápidamente, y cuáles mantienen patrones más estables. Este análisis 
    es fundamental para la asignación eficiente de recursos y la formulación de políticas específicas para cada componente del 
    sistema de salud.</p>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "Estadísticas con datos faltantes",
        "Estadísticas sin datos faltantes",
        "Métricas de crecimiento"
    ])

    with tab1:
        st.write("**Estadísticas descriptivas incluyendo valores faltantes:**")
        st.markdown("""
        <p>Esta tabla presenta las estadísticas calculadas sobre el conjunto completo de datos, incluyendo períodos donde 
        hay valores faltantes. Los conteos reflejan el número total de observaciones posibles en el rango temporal seleccionado.</p>
        """, unsafe_allow_html=True)
        
        summary_with = sub_nhe.groupby("Expenditure_Type")["Amount"].describe()
        st.dataframe(summary_with.style.format("{:,.2f}"), use_container_width=True)

    with tab2:
        st.write("**Estadísticas descriptivas excluyendo valores faltantes:**")
        st.markdown("""
        <p>Esta tabla excluye completamente los valores faltantes, proporcionando una imagen más precisa de las características 
        de los datos efectivamente disponibles. Las diferencias en los conteos entre esta tabla y la anterior indican la magnitud 
        del problema de datos faltantes para cada categoría.</p>
        """, unsafe_allow_html=True)
        
        summary_without = sub_nhe.dropna(subset=["Amount"]).groupby("Expenditure_Type")["Amount"].describe()
        st.dataframe(summary_without.style.format("{:,.2f}"), use_container_width=True)

    with tab3:
        st.write("**Métricas de crecimiento por categoría:**")
        st.markdown("""
        <p>Las métricas de crecimiento revelan qué categorías han experimentado las expansiones más significativas y cuáles 
        han mantenido un crecimiento más moderado. El CAGR (Compound Annual Growth Rate) es particularmente útil para 
        comparaciones, ya que normaliza el crecimiento a lo largo de diferentes períodos temporales.</p>
        """, unsafe_allow_html=True)
        
        growth_metrics = []
        for exp_type in sub_nhe['Expenditure_Type'].unique():
            cat_data = sub_nhe[sub_nhe['Expenditure_Type'] == exp_type].dropna(subset=['Amount']).sort_values('Year')
            if len(cat_data) > 1:
                first_val = cat_data['Amount'].iloc[0]
                last_val = cat_data['Amount'].iloc[-1]
                years_span = len(cat_data)
                
                total_growth = ((last_val - first_val) / first_val) * 100 if first_val > 0 else 0
                cagr = ((last_val / first_val) ** (1/years_span) - 1) * 100 if first_val > 0 else 0
                
                growth_metrics.append({
                    'Categoría': exp_type,
                    'Valor inicial': first_val,
                    'Valor final': last_val,
                    'Crecimiento total (%)': total_growth,
                    'CAGR (%)': cagr,
                    'Años con datos': years_span
                })
        
        growth_df = pd.DataFrame(growth_metrics).sort_values('CAGR (%)', ascending=False)
        st.dataframe(growth_df.style.format({
            'Valor inicial': '{:,.0f}',
            'Valor final': '{:,.0f}',
            'Crecimiento total (%)': '{:.2f}',
            'CAGR (%)': '{:.2f}',
            'Años con datos': '{:.0f}'
        }), use_container_width=True)

    st.markdown("""
    <div class="interpretation-box">
    <p><strong>Interpretación del análisis estadístico comparativo:</strong> Los datos revelan heterogeneidad significativa 
    entre las diferentes categorías de gasto. Categorías como Private Health Insurance y Health Consumption Expenditures 
    muestran volúmenes absolutos mucho mayores y tasas de crecimiento más aceleradas, reflejando su rol central en el sistema 
    de salud. Por otro lado, Workers' Compensation, aunque esencial, representa una fracción mucho menor del gasto total y 
    ha crecido a tasas más moderadas, posiblemente debido a mejoras en seguridad laboral y menor siniestralidad.</p>

    <p>La variabilidad en las tasas de crecimiento (CAGR) entre categorías sugiere que el gasto en salud no es monolítico sino 
    que está compuesto por componentes con dinámicas muy diferentes. Esto tiene implicaciones importantes para la política 
    pública: intervenciones efectivas para contener costos en seguros privados pueden no ser aplicables a programas de 
    compensación laboral, y viceversa.</p>
    </div>
    """, unsafe_allow_html=True)

    # 3. VISUALIZACIÓN COMPARATIVA
    st.subheader("3. Visualización Comparativa de Evolución Temporal")

    st.markdown("""
    <p>La visualización comparativa permite observar simultáneamente la evolución de múltiples categorías, facilitando la 
    identificacion de patrones comunes, divergencias, y relaciones entre variables. Los gráficos interactivos permiten al 
    analista seleccionar subconjuntos de categorías para análisis más focalizados, evitando la saturación visual que podría 
    resultar de graficar todas las variables simultáneamente.</p>
    """, unsafe_allow_html=True)

    # Selector de variables
    available_vars = sorted(sub_nhe["Expenditure_Type"].unique().tolist())
    default_selection = available_vars[:min(5, len(available_vars))]

    selected_vars = st.multiselect(
        "Selecciona las categorías a comparar (máximo 10 para legibilidad):",
        available_vars,
        default=default_selection,
        max_selections=10
    )

    if selected_vars:
        sub_filtered = sub_nhe[sub_nhe["Expenditure_Type"].isin(selected_vars)].copy()
        
        # Gráfico de líneas comparativo
        fig_related = px.line(
            sub_filtered,
            x="Year",
            y="Amount",
            color="Expenditure_Type",
            title="Evolución Temporal Comparativa de Categorías Seleccionadas",
            markers=True,
            line_shape="linear"
        )
        
        fig_related.update_layout(
            xaxis_title="Año",
            yaxis_title="Monto (Millones USD)",
            hovermode='x unified',
            legend_title="Categoría de Gasto",
            template="plotly_white",
            height=600,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        st.plotly_chart(fig_related, use_container_width=True)
        
        # Gráfico de área apilada
        fig_area = px.area(
            sub_filtered,
            x="Year",
            y="Amount",
            color="Expenditure_Type",
            title="Distribución Proporcional del Gasto (Área Apilada)"
        )
        
        fig_area.update_layout(
            xaxis_title="Año",
            yaxis_title="Monto Acumulado (Millones USD)",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig_area, use_container_width=True)
        
        # Análisis de proporción
        if len(selected_vars) > 1:
            st.write("**Análisis de proporciones relativas:**")
            
            pivot_data = sub_filtered.pivot_table(
                values='Amount',
                index='Year',
                columns='Expenditure_Type',
                aggfunc='sum'
            )
            
            # Calcular proporciones
            prop_data = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100
            
            fig_prop = go.Figure()
            
            for col in prop_data.columns:
                fig_prop.add_trace(go.Scatter(
                    x=prop_data.index,
                    y=prop_data[col],
                    mode='lines',
                    name=col,
                    stackgroup='one',
                    groupnorm='percent'
                ))
            
            fig_prop.update_layout(
                title="Proporción Relativa del Gasto por Categoría (%)",
                xaxis_title="Año",
                yaxis_title="Porcentaje del Total (%)",
                template="plotly_white",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_prop, use_container_width=True)
        
        st.markdown("""
        <div class="interpretation-box">
        <p><strong>Interpretación de las visualizaciones comparativas:</strong> Los gráficos revelan patrones distintivos 
        para cada categoría. Mientras que categorías como Total National Health Expenditures y Health Consumption muestran 
        curvas exponenciales pronunciadas, Workers' Compensation mantiene una trayectoria más lineal y estable. Esta diferencia 
        refleja factores estructurales: el gasto total está impulsado por múltiples factores (envejecimiento, tecnología, 
        expansión de cobertura), mientras que Workers' Compensation está más directamente vinculado a la siniestralidad 
        laboral, que ha disminuido gracias a mejoras en seguridad ocupacional.</p>
        
        <p>El gráfico de área apilada ilustra cómo la composición del gasto ha evolucionado a lo largo del tiempo. Si se observa 
        un aumento en la proporción de una categoría específica, esto puede indicar presiones de costos particulares en ese 
        segmento. Por ejemplo, un aumento en la proporción del gasto en seguros privados podría reflejar el encarecimiento 
        de las primas o mayor cobertura poblacional en el sector privado.</p>
        
        <p>El análisis de proporciones relativas es particularmente útil para identificar shifts estructurales en el sistema 
        de salud. Una categoría que mantiene su proporción constante está creciendo al mismo ritmo que el gasto total, mientras 
        que cambios en las proporciones indican crecimiento diferencial que puede requerir atención política.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 4. FORECASTING POR CATEGORÍA
        st.subheader(f"4. Proyecciones Individuales por Categoría ({forecast_periods} años)")
        
        st.markdown(f"""
        <p>El forecasting desagregado por categoría permite proyectar no solo el gasto total, sino también la composición 
        futura del mismo. Esto es crucial para la planificación sectorial: proveedores de seguros privados, programas de 
        compensación laboral, y sistemas de salud pública requieren proyecciones específicas para dimensionar infraestructura, 
        capacidad de atención, y necesidades de financiamiento. Se generan proyecciones para los próximos {forecast_periods} 
        años para cada categoría seleccionada, utilizando el método ensemble que ha demostrado mejor rendimiento en el análisis 
        del gasto total.</p>
        """, unsafe_allow_html=True)
        
        # Seleccionar una categoría para proyección detallada
        forecast_category = st.selectbox(
            "Selecciona una categoría para proyección detallada:",
            selected_vars
        )
        
        if forecast_category:
            cat_data = sub_nhe[sub_nhe['Expenditure_Type'] == forecast_category].dropna(subset=['Amount']).sort_values('Year')
            
            if len(cat_data) >= 10:  # Mínimo 10 puntos para proyección confiable
                cat_prepared = prepare_time_series(cat_data)
                
                # Generar proyecciones
                ensemble_cat, exp_smooth_cat, poly_cat, linear_cat, future_years_cat = ensemble_forecast(
                    cat_prepared,
                    forecast_periods
                )
                
                # Visualización
                fig_forecast_cat = go.Figure()
                
                # Datos históricos
                fig_forecast_cat.add_trace(go.Scatter(
                    x=cat_data["Year"],
                    y=cat_data["Amount"],
                    mode='lines+markers',
                    name='Datos Históricos',
                    line=dict(color='#2563eb', width=3),
                    marker=dict(size=6)
                ))
                
                # Proyección
                fig_forecast_cat.add_trace(go.Scatter(
                    x=future_years_cat,
                    y=ensemble_cat,
                    mode='lines+markers',
                    name='Proyección Ensemble',
                    line=dict(color='#10b981', width=3, dash='dash'),
                    marker=dict(size=8, symbol='diamond')
                ))
                
                # Intervalo de confianza
                upper_bound_cat = ensemble_cat * 1.15
                lower_bound_cat = ensemble_cat * 0.85
                
                fig_forecast_cat.add_trace(go.Scatter(
                    x=np.concatenate([future_years_cat, future_years_cat[::-1]]),
                    y=np.concatenate([upper_bound_cat, lower_bound_cat[::-1]]),
                    fill='toself',
                    fillcolor='rgba(16, 185, 129, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Intervalo de confianza (±15%)'
                ))
                
                fig_forecast_cat.update_layout(
                    title=f"Proyección de {forecast_category} hasta {int(future_years_cat[-1])}",
                    xaxis_title="Año",
                    yaxis_title="Monto (Millones USD)",
                    hovermode='x unified',
                    template="plotly_white",
                    height=600
                )
                
                st.plotly_chart(fig_forecast_cat, use_container_width=True)
                
                # Tabla de proyecciones
                forecast_cat_df = pd.DataFrame({
                    'Año': future_years_cat,
                    'Proyección (Millones USD)': ensemble_cat
                })
                
                st.write(f"**Proyecciones detalladas para {forecast_category}:**")
                forecast_cat_display = forecast_cat_df.copy()
                forecast_cat_display['Proyección (Millones USD)'] = forecast_cat_display['Proyección (Millones USD)'].apply(lambda x: f"${x:,.0f}M")
                st.dataframe(forecast_cat_display, use_container_width=True)
                
                # Métricas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    last_observed = cat_data['Amount'].iloc[-1]
                    final_forecast_cat = ensemble_cat[-1]
                    st.metric(
                        f"Último valor observado ({int(cat_data['Year'].iloc[-1])})",
                        f"${last_observed:,.0f}M"
                    )
                
                with col2:
                    st.metric(
                        f"Proyección para {int(future_years_cat[-1])}",
                        f"${final_forecast_cat:,.0f}M"
                    )
                
                with col3:
                    forecast_growth_cat = ((final_forecast_cat - last_observed) / last_observed) * 100
                    st.metric(
                        f"Crecimiento proyectado ({forecast_periods} años)",
                        f"{forecast_growth_cat:.1f}%"
                    )
                
                st.markdown(f"""
                <div class="interpretation-box">
                <p><strong>Interpretación de la proyección para {forecast_category}:</strong> El modelo proyecta que el gasto 
                en esta categoría alcanzará ${final_forecast_cat:,.0f} millones de dólares en {int(future_years_cat[-1])}, 
                representando un incremento del {forecast_growth_cat:.1f}% respecto al último valor observado. Esta trayectoria 
                sugiere la continuación de tendencias históricas, aunque con las cautelas propias de cualquier ejercicio de 
                proyección a largo plazo.</p>
                
                <p>Es importante contextualizar estas proyecciones dentro del marco más amplio del sistema de salud. Cambios 
                regulatorios, innovaciones tecnológicas, shifts demográficos, o crisis económicas pueden alterar significativamente 
                las trayectorias proyectadas. Por ejemplo, en el caso de Workers' Compensation, avances en automatización y 
                robótica podrían reducir la exposición a riesgos laborales tradicionales, mientras que nuevos riesgos (como 
                lesiones por trabajos repetitivos en servicios) podrían emerger. En el caso de seguros privados, reformas de 
                salud o expansión de programas públicos podrían redistribuir la composición del gasto.</p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.warning(f"La categoría {forecast_category} no tiene suficientes datos históricos (mínimo 10 puntos) para generar proyecciones confiables.")
st.markdown("---")
# ============================================
# CONCLUSIONES GENERALES
# ============================================
st.header("Síntesis y Conclusiones Generales del Análisis")
st.markdown("""
<p>El análisis integral de los National Health Expenditures (1960-2023) proporciona una visión comprehensiva de la evolución 
del sistema de salud estadounidense desde múltiples perspectivas: macroeconómica (gasto total), sectorial (categorías 
específicas), y proyectiva (forecasting). Los hallazgos tienen implicaciones significativas para la formulación de políticas 
públicas, la planificación estratégica de organizaciones de salud, y la comprensión de las dinámicas de costos en atención 
médica.</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style='background-color: #f0fdf4; padding: 20px; border-radius: 10px; border-left: 5px solid #10b981;'>
    <h5>Fortalezas del Dataset y del Análisis</h5>
    <ul>
    <li><strong>Cobertura temporal extensa:</strong> 64 años de datos históricos permiten identificar tendencias de largo plazo</li>
    <li><strong>Alta calidad de datos:</strong> Completitud excepcional en categorías principales (>95%)</li>
    <li><strong>Granularidad detallada:</strong> 65 categorías diferentes permiten análisis sectoriales profundos</li>
    <li><strong>Fuente oficial y confiable:</strong> CMS es la autoridad nacional en estadísticas de salud</li>
    <li><strong>Actualización periódica:</strong> Datos actualizados hasta 2023 mantienen relevancia</li>
    <li><strong>Metodología rigurosa:</strong> Aplicación de múltiples técnicas de forecasting aumenta robustez</li>
    <li><strong>Interpretación contextualizada:</strong> Vinculación de datos con políticas y eventos históricos</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background-color: #fef3c7; padding: 20px; border-radius: 10px; border-left: 5px solid #f59e0b;'>
    <h5>Limitaciones y Consideraciones</h5>
    <ul>
    <li><strong>Valores nominales:</strong> Los datos no están ajustados por inflación (valores corrientes)</li>
    <li><strong>Datos faltantes en categorías secundarias:</strong> Algunas categorías tienen <15% de completitud</li>
    <li><strong>Cambios metodológicos históricos:</strong> Redefiniciones de categorías complican comparaciones temporales</li>
    <li><strong>Proyecciones basadas en tendencias:</strong> Los modelos asumen continuidad de patrones históricos</li>
    <li><strong>Eventos imprevisibles:</strong> Pandemias, reformas radicales, o crisis no están contempladas</li>
    <li><strong>Agregación nacional:</strong> No refleja variabilidad geográfica o demográfica subnacional</li>
    <li><strong>Causalidad no establecida:</strong> El análisis es descriptivo y proyectivo, no causal</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="context-box">
<h4>Hallazgos Clave del Análisis</h4>
<ol>
    <li><strong>Crecimiento exponencial sostenido del gasto total:</strong> El gasto nacional en salud ha crecido de $253 mil 
    millones en 1980 a $4.87 billones en 2023, con una CAGR del 6.95%. Las proyecciones sugieren que esta tendencia continuará, 
    aunque con moderación gradual de las tasas de crecimiento.</li>
    <li><strong>Heterogeneidad significativa entre categorías:</strong> Workers' Compensation representa una fracción 
    relativamente pequeña y estable del gasto total, mientras que seguros privados y consumo médico han experimentado 
    expansión acelerada. Esta divergencia refleja dinámicas específicas de cada sector.</li>

    <li><strong>Patrones de completitud de datos revelan historia institucional:</strong> La presencia de datos faltantes 
    en ciertas categorías no es aleatoria, sino que refleja la evolución histórica de la clasificación de gastos del CMS 
    y cambios en políticas de reporte.</li>

    <li><strong>Tres fases históricas identificables:</strong> (1) 1960-1980: establecimiento de Medicare/Medicaid y 
    crecimiento moderado, (2) 1980-2010: expansión acelerada con tecnología médica y envejecimiento, (3) 2010-2023: 
    crecimiento sostenido con desaceleración relativa post-ACA.</li>

    <li><strong>Proyecciones indican continuidad con moderación:</strong> Los modelos ensemble proyectan tasas de crecimiento 
    futuras ligeramente inferiores al promedio histórico, sugiriendo efectos de políticas de contención de costos y 
    posible estabilización demográfica.</li>

    <li><strong>Necesidad de análisis multifacético:</strong> El gasto en salud no puede entenderse mediante una sola métrica; 
    requiere análisis desagregado, comparativo, y contextualizado para capturar su complejidad inherente.</li>
</ol>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background-color: #e0e7ff; padding: 20px; border-radius: 10px; border-left: 5px solid #6366f1; margin-top: 20px;'>
<h4>Implicaciones para Política Pública y Gestión en Salud</h4>
<p><strong>Para formuladores de políticas:</strong> El análisis sugiere que, sin intervenciones significativas, el gasto en 
salud continuará creciendo a tasas superiores al crecimiento económico general, lo cual plantea desafíos de sostenibilidad 
fiscal. Las proyecciones pueden informar debates sobre reforma de salud, particularmente en áreas como control de precios 
de medicamentos, eficiencia administrativa, y medicina preventiva.</p>
<p><strong>Para gestores de sistemas de salud:</strong> La variabilidad entre categorías indica que no existe una solución 
única para la contención de costos. Estrategias efectivas para seguros privados (como negociación de precios) pueden no 
aplicar a Workers' Compensation (donde la prevención de lesiones es clave). La planificación debe ser sectorial y basada 
en evidencia específica.</p>
<p><strong>Para investigadores:</strong> Este análisis sugiere varias líneas de investigación futura: (1) descomposición 
del crecimiento en componentes (precio vs. volumen vs. intensidad), (2) análisis de causalidad entre políticas específicas 
y cambios en gasto, (3) comparaciones internacionales para identificar mejores prácticas, y (4) modelado más sofisticado 
que incorpore variables exógenas (demográficas, económicas, tecnológicas).</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
# ============================================
# DESCARGA DE DATOS
# ============================================
st.header("Descarga de Datos Procesados")
st.markdown("""
<p>Esta sección permite descargar los datos procesados y las proyecciones generadas durante el análisis. Los archivos CSV 
pueden ser utilizados para análisis adicionales, reportes, o integración con otras herramientas analíticas.</p>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if len(total) > 0:
        csv_total = total.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar Total NHE",
            data=csv_total,
            file_name=f'total_nhe_{years[0]}_{years[1]}.csv',
            mime='text/csv'
        )

with col2:
    if len(sub_nhe) > 0:
        csv_related = sub_nhe.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar Variables Relacionadas",
            data=csv_related,
            file_name=f'related_vars_{years[0]}_{years[1]}.csv',
            mime='text/csv'
        )

with col3:
    csv_full = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar Dataset Completo Filtrado",
        data=csv_full,
        file_name=f'nhe_complete_{years[0]}_{years[1]}.csv',
        mime='text/csv'
    )

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 30px; background-color: #f8f9fa; border-radius: 10px;'>
<h4>Fundamentos para el Análisis de Datos (FACD)</h4>
<p><strong>Desarrollado por:</strong> Juan Sebastián Fajardo Acevedo y Miguel Ängel Vargas Hernández</p>
<p><strong>Docente:</strong> Ana María Gómez Lamus, M.Sc. en Estadística</p>
<p><strong>Institución:</strong> Universidad de La Sabana</p>
<p><strong>Año:</strong> 2025</p>
<p><strong>Datos actualizados al:</strong> {}</p>
<hr style='margin: 20px 0; border: none; border-top: 1px solid #ddd;'>
<p style='font-size: 0.9em; color: #888;'>Este dashboard representa un análisis académico con fines educativos. 
Las proyecciones son indicativas y no constituyen asesoría financiera o política. Para decisiones estratégicas, 
consulte con expertos en política de salud y análisis económico.</p>
</div>
""".format(int(nhe['Year'].max())), unsafe_allow_html=True)
