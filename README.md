Previsão e Análise de Obesidade com Machine Learning

Este projeto contém aplicações desenvolvidas com Streamlit para previsão de obesidade em pacientes e análise exploratória dos resultados. O modelo de Machine Learning utilizado é baseado em Random Forest.

Arquivos do Projeto:

1) pipeline_random_forest.pkl:
   Modelo de Machine Learning já treinado e pronto para uso.

2) app.py:
   Aplicação para inserção de dados do paciente e previsão da categoria de obesidade.
   Salva automaticamente cada previsão em um arquivo predicoes.csv.

3) dashboard_resultados.py:
   Dashboard interativo para análise das previsões salvas. Exibe métricas gerais, gráfico de distribuição das classes e estatísticas descritivas.

4) predicoes.csv:
   Arquivo gerado automaticamente pelo app.py contendo o histórico de previsões (necessário para o dashboard).

5) requirements.txt:
   Lista das dependências necessárias para executar as aplicações.


