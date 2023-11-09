import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import rmse
def roe_forecast():

    # Caminho do arquivo de dados da ação
    stock_path = 'Data/VALE3.csv'
    print('Time series forecast for ' + stock_path)

    # Carrega os dados do arquivo CSV
    df = pd.read_csv(stock_path)
    df.head()

    validation_size = 2

    # Dados para treinamento =  DataFrame - tamanho da validação
    training_df = df.head(len(df) - validation_size)
    print("Training: ", training_df)

    validation_df = df.tail(validation_size)

    # Cria uma instância do modelo Prophet
    m = Prophet()

    # Ajusta o modelo aos dados
    m.fit(training_df)

    # Gera datas futuras para previsão
    future = m.make_future_dataframe(periods=len(validation_df), freq='Q')
    print("DataFrame futuro \n", future.tail())

    # Realiza a previsão
    forecast = m.predict(future)
    print(forecast.tail())

    # Exibe os resultados da previsão
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    predictions = forecast.tail(len(validation_df))

    print("Predictions: ", predictions)

    print("Validation: ", validation_df)

    # rmse entre o conjunto de dados previstos e o conjunto completo
    print("RMSE completo: ", rmse(forecast['yhat'], df['y']))

    #rmse entre os dados de validação e e os dados previstos
    print("RMSE predito: ", rmse(predictions['yhat'],validation_df['y']))

    # Plota o gráfico da previsão
    fig1 = m.plot(forecast)

    # Plota componentes da previsão
    fig2 = m.plot_components(forecast)

    # Exibe os gráficos
    plt.show()
