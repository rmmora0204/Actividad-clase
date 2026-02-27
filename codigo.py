import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import io
import csv
import numpy as np

# --- Configuración de la página Streamlit ---
st.set_page_config(page_title="Dashboard de Delitos en Barranquilla", layout="wide")
st.title("Análisis de Delitos de Alto Impacto en Barranquilla")

# --- Función para cargar y limpiar los datos (reutilizada y adaptada) ---
@st.cache_data
def load_and_clean_data(file_path):
    encoding = 'latin1'
    processed_rows = []

    with open(file_path, 'r', encoding=encoding) as f:
        raw_content = f.read()

    for raw_line in raw_content.splitlines():
        line = raw_line.strip()
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        line = line.replace('""', '"')

        reader = csv.reader(io.StringIO(line), delimiter=',', quotechar='"')
        try:
            processed_rows.append(next(reader))
        except StopIteration:
            continue

    if not processed_rows:
        raise ValueError("No data could be processed from the file.")

    headers = [h.replace('"', '').replace('Ã±', 'n').replace('Ãº', 'u').replace('  ', ' ').strip() for h in processed_rows[0]]
    df = pd.DataFrame(processed_rows[1:], columns=headers)

    # Convert numeric columns
    numeric_cols = [
        'Casos/denuncias anterior periodo',
        'Casos/denuncias ultimo periodo',
        'Variacion absoluta'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('"', '').str.replace(',', ''), errors='coerce')

    if 'Variacion %' in df.columns:
        df['Variacion %'] = df['Variacion %'].astype(str).str.replace('%', '', regex=False)
        df['Variacion %'] = pd.to_numeric(df['Variacion %'], errors='coerce')

    # Categorize 'Delito' column
    def categorize_delito(delito):
        delito_lower = str(delito).lower()
        if 'homicidios' in delito_lower: return 'homicidios'
        elif 'lesiones' in delito_lower: return 'lesiones_personales'
        elif 'hurto' in delito_lower: return 'hurto'
        elif 'extorsi' in delito_lower: return 'extorsion'
        elif 'violencia intrafamiliar' in delito_lower: return 'violencia_intrafamiliar'
        elif 'delito sexual' in delito_lower or 'delitos sexuales' in delito_lower: return 'delito_sexual'
        else: return 'otros'

    df['Categoria_Delito'] = df['Delito'].apply(categorize_delito)

    return df

# --- Carga de datos ---
# Make sure the CSV file is accessible in the Colab environment
file_path = '/content/Comparativo_de_delitos_de_alto_impacto_en_la_ciudad_de_Barranquilla_20260221.csv'
try:
    df_delitos = load_and_clean_data(file_path)
    st.success("Datos cargados y procesados correctamente.")
except Exception as e:
    st.error(f"Error al cargar o procesar los datos: {e}")
    st.stop() # Stop the app if data loading fails

# --- Pestañas para las gráficas ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Distribución por Categoría",
    "Correlación Violencia/Sexual",
    "Serie de Tiempo por Categoría",
    "Hurtos vs. Lesiones",
    "Comparación de Denuncias"
])

# --- Gráfica 1: Distribución de Categoria_Delito ---
with tab1:
    st.header("1. Distribución de Delitos por Categoría")
    category_counts = df_delitos['Categoria_Delito'].value_counts().reset_index()
    category_counts.columns = ['Categoria_Delito', 'Count']
    category_counts['Percentage'] = (category_counts['Count'] / category_counts['Count'].sum()) * 100
    category_counts = category_counts.sort_values(by='Count', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Count', y='Categoria_Delito', data=category_counts, palette='viridis', ax=ax)
    for index, row in category_counts.iterrows():
        ax.text(row['Count'] + 5, index, f"{row['Count']} ({row['Percentage']:.1f}%)", color='black', ha="left", va="center")
    ax.set_title('Distribución de Categorías de Delito')
    ax.set_xlabel('Conteo de Casos')
    ax.set_ylabel('Tipo de Delito')
    plt.tight_layout()
    st.pyplot(fig)

# --- Gráfica 2: Heatmap de correlación entre 'violencia_intrafamiliar' y 'delito_sexual' ---
with tab2:
    st.header("2. Correlación entre Violencia Intrafamiliar y Delito Sexual")
    df_filtered_corr = df_delitos[df_delitos['Categoria_Delito'].isin(['violencia_intrafamiliar', 'delito_sexual'])]

    # Ensure column names are correct after cleaning
    anos_col = [col for col in df_delitos.columns if 'Anos comparados' == col][0]
    periodo_meses_col = [col for col in df_delitos.columns if 'Periodo meses comparado' == col][0]
    casos_ultimo_col = [col for col in df_delitos.columns if 'Casos/denuncias ultimo periodo' == col][0]

    df_pivot = df_filtered_corr.pivot_table(
        index=[periodo_meses_col, anos_col],
        columns='Categoria_Delito',
        values=casos_ultimo_col,
        aggfunc='sum'
    ).fillna(0)

    # Check if both columns exist in df_pivot before correlating
    corr_cols = [col for col in ['violencia_intrafamiliar', 'delito_sexual'] if col in df_pivot.columns]
    if len(corr_cols) == 2:
        correlation_table = df_pivot[corr_cols].corr(method='pearson')
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(correlation_table, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=ax)
        ax.set_title('Mapa de Calor de Correlación: Violencia Intrafamiliar vs. Delito Sexual')
        st.pyplot(fig)
    else:
        st.warning("No hay suficientes datos para calcular la correlación entre violencia intrafamiliar y delito sexual.")

# --- Gráfica 3: Serie de tiempo de todas las categorías de delito ---
with tab3:
    st.header("3. Serie de Tiempo de Todas las Categorías de Delito")
    anos_col = [col for col in df_delitos.columns if 'Anos comparados' == col][0]
    casos_ultimo_col = [col for col in df_delitos.columns if 'Casos/denuncias ultimo periodo' == col][0]

    time_series_data = df_delitos.groupby([anos_col, 'Categoria_Delito'])[casos_ultimo_col].sum().reset_index()
    time_series_data = time_series_data[time_series_data['Categoria_Delito'] != 'otros']

    fig = px.line(
        time_series_data,
        x=anos_col,
        y=casos_ultimo_col,
        color='Categoria_Delito',
        markers=True,
        title='Serie de Tiempo de Todas las Categorías de Delito',
        labels={casos_ultimo_col: 'Total de Casos', anos_col: 'Años Comparados'}
    )
    fig.update_xaxes(type='category', categoryorder='category ascending')
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig)

# --- Gráfica 4: Comparación de casos totales: Hurtos vs. Lesiones Personales ---
with tab4:
    st.header("4. Comparación de Casos Totales: Hurtos vs. Lesiones Personales")
    casos_ultimo_col = [col for col in df_delitos.columns if 'Casos/denuncias ultimo periodo' == col][0]

    comparison_data_bar = df_delitos[df_delitos['Categoria_Delito'].isin(['hurto', 'lesiones_personales'])]
    aggregated_comparison_data = comparison_data_bar.groupby('Categoria_Delito')[casos_ultimo_col].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Categoria_Delito', y=casos_ultimo_col, data=aggregated_comparison_data, palette='coolwarm', ax=ax)
    for index, row in aggregated_comparison_data.iterrows():
        ax.text(index, row[casos_ultimo_col] + 100, f"{row[casos_ultimo_col]:.0f}", color='black', ha="center", va="bottom")
    ax.set_title('Comparación de Casos Totales: Hurtos vs. Lesiones Personales')
    ax.set_xlabel('Categoría de Delito')
    ax.set_ylabel('Total de Casos')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig)

# --- Gráfica 5: Gráfico de barras agrupadas de denuncias por categoría y período ---
with tab5:
    st.header("5. Comparación de Denuncias y Total de Delitos por Categoría")
    casos_anterior_col = [col for col in df_delitos.columns if 'Casos/denuncias anterior periodo' == col][0]
    casos_ultimo_col = [col for col in df_delitos.columns if 'Casos/denuncias ultimo periodo' == col][0]

    agregated_denuncias_by_category = df_delitos.groupby('Categoria_Delito')[[casos_anterior_col, casos_ultimo_col]].sum().reset_index()
    agregated_denuncias_by_category['Total Casos'] = agregated_denuncias_by_category[casos_anterior_col] + agregated_denuncias_by_category[casos_ultimo_col]

    df_melted_comparison = agregated_denuncias_by_category.melt(
        id_vars='Categoria_Delito',
        value_vars=[casos_anterior_col, casos_ultimo_col, 'Total Casos'],
        var_name='Tipo de Periodo',
        value_name='Numero de Casos'
    )
    df_melted_comparison['Tipo de Periodo'] = df_melted_comparison['Tipo de Periodo'].replace({
        casos_anterior_col: 'Periodo Anterior',
        casos_ultimo_col: 'Último Periodo',
        'Total Casos': 'Total Delitos'
    })

    fig, ax = plt.subplots(figsize=(15, 9))
    sns.barplot(x='Numero de Casos', y='Categoria_Delito', hue='Tipo de Periodo', data=df_melted_comparison, palette='dark', ax=ax)
    ax.set_title('Comparación de Denuncias y Total de Delitos por Categoría')
    ax.set_xlabel('Número de Casos')
    ax.set_ylabel('Categoría de Delito')
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.legend(title='Tipo de Periodo')
    plt.tight_layout()
    st.pyplot(fig)
