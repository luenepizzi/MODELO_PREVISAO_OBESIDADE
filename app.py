import streamlit as st
import pandas as pd
import pickle
import os


def feature_engineering(df):
    df["IMC"] = df["Weight"] / (df["Height"] ** 2)
    return df

# Carregar modelo

with open("pipeline_random_forest.pkl", "rb") as f:
    pipeline = pickle.load(f)

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("""
<style>
    /* Cor geral dos títulos */
    h1, h2, h3 {
        color: #ff69b4;
    }
    /* Caixa customizada para avisos */
    .pink-box {
        padding: 10px;
        border-radius: 10px;
        background-color: #ffb6c1;
        color: #8b0057;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
    }
    /* Caixa para destaque principal */
    .pink-highlight {
        padding: 10px;
        border-radius: 10px;
        background-color: #ff69b4;
        color: white;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# Função para previsão utilizando dados do paciente

def prever_paciente(dados):
    df = pd.DataFrame([dados])
    pred = pipeline.predict(df)[0]
    return pred

# Interface Streamlit

st.title("Previsão de Obesidade")

st.write("""
Insira os dados do paciente para prever a categoria de obesidade:
""")

# Formulário
gender = st.selectbox("Gênero", ["Male", "Female"])
age = st.number_input("Idade", 1, 120, 25)
height = st.number_input("Altura (m)", 1.0, 2.5, 1.7, step=0.01)
weight = st.number_input("Peso (kg)", 20, 250, 70, step=1)
family_history = st.selectbox("Tem histórico familiar de obesidade?", ["yes", "no"])
favc = st.selectbox("Consome alimentos calóricos com frequência?", ["yes", "no"])
fcvc = st.slider("Costuma comer vegetais nas refeições?", 1.0, 3.0, 2.0)
ncp = st.slider("Qual é o número de refeições que faz por dia?", 1, 5, 3)
caec = st.selectbox("Come alguma coisa entre refeições?", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Fuma?", ["yes", "no"])
ch2o = st.slider("Quanta água ingere diariamente? (L/dia)", 1.0, 3.0, 2.0)
scc = st.selectbox("Monitora calorias ingeridas?", ["yes", "no"])
faf = st.slider("Com que frequência pratica atividade física? (horas por semana)", 0.0, 5.0, 1.0)
tue = st.slider("Quanto tempo passa em dispositivos eletrônicos? (horas por dia)", 0.0, 3.0, 1.0)
calc = st.selectbox("Com que frequência costuma ingerir álcool?", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Qual meio de transaporte mais utiliza?", ["Walking", "Public_Transportation", "Automobile", "Motorbike", "Bike"])

# Botão de previsão
if st.button("Prever"):
    dados_paciente = {
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "family_history": family_history,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans
    }

    resultado = prever_paciente(dados_paciente)
    
    # Caixa de destaque para resultado
    st.markdown(f"<div class='pink-highlight'>Classe prevista: {resultado}</div>", unsafe_allow_html=True)


    # Salvar previsão no CSV
    novo_paciente_df = pd.DataFrame([dados_paciente])
    novo_paciente_df["Predição_Obesidade"] = resultado

    arquivo_predicoes = "predicoes.csv"
    if not os.path.exists(arquivo_predicoes):
        novo_paciente_df.to_csv(arquivo_predicoes, index=False)
    else:
        novo_paciente_df.to_csv(arquivo_predicoes, mode='a', header=False, index=False)

    st.markdown("<div class='pink-box'>Previsão salva! Confira no dashboard de insights.</div>", unsafe_allow_html=True)
