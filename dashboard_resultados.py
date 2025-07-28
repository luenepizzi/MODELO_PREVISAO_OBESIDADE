import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
import numpy as np
import os

# Carregar modelo treinado
with open("pipeline_random_forest.pkl", "rb") as f:
    pipeline = pickle.load(f)

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Previsão de Obesidade - Resultados")

st.write("""
Este dashboard apresenta os resultados do Modelo de Previsão de Obesidade.
""")

# Ler arquivo CSV gerado pelo app
arquivo_predicoes = "predicoes.csv"

if os.path.exists(arquivo_predicoes):
    df = pd.read_csv(arquivo_predicoes)

    # Criar coluna IMC
    if "Weight" in df.columns and "Height" in df.columns:
        df["IMC"] = df["Weight"] / (df["Height"] ** 2)

    # Filtros interativos
    st.sidebar.header("Filtros")

    # Filtro de gênero
    if "Gender" in df.columns:
        generos = df["Gender"].unique().tolist()
        genero_selecionado = st.sidebar.multiselect(
            "Filtrar por gênero",
            options=generos,
            default=generos
        )
    else:
        genero_selecionado = []

    # Filtro de tipo de obesidade
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

    # ---------------------------
    # Visualização inicial
    # ---------------------------
    st.subheader("Overview")
    st.dataframe(df_filtrado.tail(10))

    # ---------------------------
    # Distribuição das classes previstas
    # ---------------------------
    st.subheader("Distribuição das Classes Previstas")
    fig, ax = plt.subplots(figsize=(8,4))
    df_filtrado["Predição_Obesidade"].value_counts().plot(kind="bar", ax=ax)
    plt.title("Distribuição das Classes de Obesidade")
    plt.xlabel("Classe")
    plt.ylabel("Quantidade")
    st.pyplot(fig)

    # Boxplot do IMC por classe
    if "IMC" in df_filtrado.columns:
        st.subheader("Boxplot do IMC por Classe Prevista")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.boxplot(x="Predição_Obesidade", y="IMC", data=df_filtrado, ax=ax)
        plt.title("Distribuição do IMC por Classe Prevista")
        st.pyplot(fig)

    # Correlação Idade x IMC
    if "Age" in df_filtrado.columns and "IMC" in df_filtrado.columns:
        st.subheader("Correlação Idade x IMC por Classe Prevista")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.scatterplot(x="Age", y="IMC", hue="Predição_Obesidade", data=df_filtrado, ax=ax)
        plt.title("Correlação Idade x IMC por Classe Prevista")
        st.pyplot(fig)

    # Importância das variáveis com SHAP
    st.subheader("Importância Global das Variáveis (SHAP)")

    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    df_processed = pipeline.named_steps["preprocess"].transform(pipeline.named_steps["feature_eng"].transform(df_filtrado))
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()

    shap_values = explainer.shap_values(df_processed)
    shap.summary_plot(np.mean(np.abs(shap_values), axis=0), pd.DataFrame(df_processed, columns=feature_names))
    st.pyplot()

    # Estatísticas descritivas
    st.subheader("Estatísticas Descritivas")
    st.write(df_filtrado.describe(include='all'))

else:
    st.warning("Nenhum dado encontrado! Inclua os dados dos seus pacientes no APP - Modelo de Previsão de Obesidade.")
