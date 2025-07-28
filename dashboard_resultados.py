import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os


# Estilos CSS

st.markdown("""
<style>
    h1, h2, h3 { color: #ff69b4; }
    [data-testid="stSidebar"] {
        background-color: #ffe4e9;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label, [data-testid="stSidebar"] div {
        color: #ff69b4;
    }
</style>
""", unsafe_allow_html=True)


# Título

st.markdown("<h1>Previsão de Obesidade - Resultados</h1>", unsafe_allow_html=True)


# Carregar dados

arquivo_predicoes = "predicoes.csv"

if os.path.exists(arquivo_predicoes):
    df = pd.read_csv(arquivo_predicoes)

    # Criar IMC
    if "Weight" in df.columns and "Height" in df.columns:
        df["IMC"] = df["Weight"] / (df["Height"] ** 2)

    
    # Filtros
    
    st.sidebar.header("Filtros")

    generos = df["Gender"].unique().tolist() if "Gender" in df.columns else []
    genero_selecionado = st.sidebar.multiselect(
        "Filtrar por gênero",
        options=generos,
        default=generos
    )

    tipos_obesidade = df["Predição_Obesidade"].unique().tolist()
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
        df_filtrado = df_filtrado[df_filtrado["Predição_Obesidade"].isin(tipo_selecionado)]

    
    # Cards de métricas
    
    total_pacientes = len(df_filtrado)
    obesidade_confirmada = df_filtrado["Predição_Obesidade"].isin(
        ["Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]
    ).sum()
    media_peso = df_filtrado["Weight"].mean() if "Weight" in df_filtrado.columns else 0
    media_altura = df_filtrado["Height"].mean() if "Height" in df_filtrado.columns else 0
    media_idade = df_filtrado["Age"].mean() if "Age" in df_filtrado.columns else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total de Pacientes", f"{total_pacientes}")
    col2.metric("Obesidade Confirmada", f"{obesidade_confirmada}")
    col3.metric("Peso Médio (kg)", f"{media_peso:.1f}")
    col4.metric("Altura Média (m)", f"{media_altura:.2f}")
    col5.metric("Idade Média (anos)", f"{media_idade:.1f}")

    
    # Gráfico de distribuição
    
    st.subheader("Distribuição das Classes Previstas")
    fig, ax = plt.subplots(figsize=(8,4))
    df_filtrado["Predição_Obesidade"].value_counts().plot(
        kind="bar", ax=ax, color="#ff69b4"
    )
    plt.title("Distribuição das Classes de Obesidade", color="#ff69b4", fontsize=14)
    plt.xlabel("Classe", color="#ff69b4")
    plt.ylabel("Quantidade", color="#ff69b4")
    st.pyplot(fig)

    
    # Estatísticas descritivas - apenas variáveis numéricas
    st.subheader("Estatísticas Descritivas")
    st.write(df_filtrado.describe(include='number'))


    
    # Tabela Completa
    
    st.subheader("Tabela Anaíltica - Informações dos Pacientes")
    st.dataframe(df_filtrado)

else:
    st.warning("Nenhum dado encontrado! Inclua as informações dos seus pacientes no aplicativo primeiro.")
