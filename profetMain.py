import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt


def main():
    # Inicializa a análise de previsão de séries temporais
    print('Init Fundamentalist Analysis Forecast')

    # Caminho do arquivo de dados da ação
    stock_path = 'Data/VALE3.csv'
    print('Time series forecast for ' + stock_path)

    # Carrega os dados do arquivo CSV
    df = pd.read_csv(stock_path)
    df.head()

    # Cria uma instância do modelo Prophet
    m = Prophet()

    # Ajusta o modelo aos dados
    m.fit(df)

    # Gera datas futuras para previsão
    future = m.make_future_dataframe(periods=6, freq='Q')
    print(future.tail())

    # Realiza a previsão
    forecast = m.predict(future)
    print(forecast.tail())

    # Exibe os resultados da previsão
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Plota o gráfico da previsão
    fig1 = m.plot(forecast)

    # Plota componentes da previsão
    fig2 = m.plot_components(forecast)

    # Plota o gráfico interativo usando Plotly
    plot_plotly(m, forecast)

    # Exibe os gráficos
    plt.show()


if __name__ == '__main__':
    main()
