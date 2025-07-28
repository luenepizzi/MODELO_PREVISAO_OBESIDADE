import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import os


# Estilos CSS

st.markdown("""
<style>
    /* Títulos e sidebar */
    h1, h2, h3 { color: #ff69b4; }
    [data-testid="stSidebar"] {
        background-color: #ffe4e9;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label, [data-testid="stSidebar"] div {
        color: #ff69b4;
    }

    /* Chips (elementos selecionados nos multiselect) */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #c71585 !important; /* rosa escuro */
        color: white !important;
        border-radius: 8px;
    }

    /* Botões de remoção dos chips (o "x") */
    .stMultiSelect [data-baseweb="tag"] svg {
        fill: white !important;
    }
</style>
""", unsafe_allow_html=True)



# Função para criar seção de dashboard

def dashboard_section(df, titulo):

    # Criar IMC
    if "Weight" in df.columns and "Height" in df.columns:
        df["IMC"] = df["Weight"] / (df["Height"] ** 2)

    # Filtros
    st.sidebar.header(f"Filtros - {titulo}")

    generos = df["Gender"].unique().tolist() if "Gender" in df.columns else []
    genero_selecionado = st.sidebar.multiselect(
        "Filtrar por gênero",
        options=generos,
        default=generos
    )

    tipos_obesidade = df["Predicao_Obesidade"].unique().tolist()
    tipo_selecionado = st.sidebar.multiselect(
        "Filtrar por tipo de obesidade",
        options=tipos_obesidade,
        default=tipos_obesidade
    )

    # Aplicar filtros
    df_filtrado = df.copy()
    if genero_selecionado:
        df_filtrado = df_filtrado[df_filtrado["Gender"].isin(genero_selecionado)]
    if tipo_selecionado:
        df_filtrado = df_filtrado[df_filtrado["Predicao_Obesidade"].isin(tipo_selecionado)]

    # Cards de métricas
    total_pacientes = len(df_filtrado)
    obesidade_confirmada = df_filtrado["Predicao_Obesidade"].isin(
        ["Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]
    ).sum()
    media_peso = df_filtrado["Weight"].mean() if "Weight" in df_filtrado.columns else 0
    media_altura = df_filtrado["Height"].mean() if "Height" in df_filtrado.columns else 0
    media_idade = df_filtrado["Age"].mean() if "Age" in df_filtrado.columns else 0
    media_imc = df_filtrado["IMC"].mean() if "IMC" in df_filtrado.columns else 0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total de Pacientes", f"{total_pacientes}")
    col2.metric("Obesidade Confirmada", f"{obesidade_confirmada}")
    col3.metric("Peso Médio (kg)", f"{media_peso:.1f}")
    col4.metric("Altura Média (m)", f"{media_altura:.2f}")
    col5.metric("Idade Média (anos)", f"{media_idade:.1f}")
    col6.metric("IMC Médio", f"{media_imc:.2f}")

    # Gráfico de distribuição
    st.subheader("Distribuição das Classes Previstas")
    fig, ax = plt.subplots(figsize=(8,4))
    df_filtrado["Predicao_Obesidade"].value_counts().plot(
        kind="bar", ax=ax, color="#ff69b4"
    )
    plt.title("Distribuição das Classes de Obesidade", color="#ff69b4", fontsize=14)
    plt.xlabel("Classe", color="#ff69b4")
    plt.ylabel("Quantidade", color="#ff69b4")
    st.pyplot(fig)

    # Estatísticas descritivas - apenas variáveis numéricas
    st.subheader("Estatísticas Descritivas")
    st.write(df_filtrado.describe(include='number'))

    # Cálculo de correlação com variável-alvo
    df_corr = df_filtrado.copy()
    class_mapping = {cls: i for i, cls in enumerate(df_corr["Predicao_Obesidade"].unique())}
    df_corr["Obesidade_Num"] = df_corr["Predicao_Obesidade"].map(class_mapping)

    numeric_cols = df_corr.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols.remove("Obesidade_Num")

    correlations = df_corr[numeric_cols].corrwith(df_corr["Obesidade_Num"]).abs().sort_values(ascending=False)
    top5_vars = correlations.head(5)

    st.subheader("Top 5 Variáveis mais Correlacionadas com Obesidade")
    st.table(top5_vars.reset_index().rename(columns={"index": "Variável", 0: "Correlação Absoluta"}))

    # Tabela Completa
    st.subheader("Tabela Analítica - Informações dos Pacientes")
    st.dataframe(df_filtrado)



# Título principal

st.markdown("<h1>Previsão de Obesidade</h1>", unsafe_allow_html=True)


# Abas

aba1, aba2 = st.tabs(["Resultados Previstos", "Base Original"])

# Aba 1 - Predições
with aba1:
    arquivo_predicoes = "predicoes.csv"
    if os.path.exists(arquivo_predicoes):
        df_pred = pd.read_csv(arquivo_predicoes)
        dashboard_section(df_pred, "Predições")
    else:
        st.warning("Nenhum dado encontrado! Inclua as informações dos seus pacientes no aplicativo de previsão.")

# Aba 2 - Base Original
with aba2:
    arquivo_original = "Obesity.csv"
    if os.path.exists(arquivo_original):
        df_original = pd.read_csv(arquivo_original)
        dashboard_section(df_original, "Base Original")
    else:
        st.warning("Nenhuma base original encontrada! Verifique se o arquivo 'Obesity.csv' está na pasta do app.")

