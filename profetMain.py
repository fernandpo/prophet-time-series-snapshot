import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt



def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    print_hi('Init Fundamentalist Analysis Forcast')
    stock_path = 'Data/VALE3.csv'
    print('Time series forecast for' + stock_path)

    df = pd.read_csv(stock_path)
    df.head()

    m = Prophet()
    train_set = df.iloc[:-10]
    print('training set')
    print(train_set)
    teste_set = df.iloc[-5:]
    print('teste set')
    print(teste_set)
    m.fit(train_set)

    # Analise dos erros -> RMSE
    # Medidas de erros series temporais
    # Prophet X Pycaret;
    # Prophet X Arima (Jupter notebook);
    # Prophet X Sarima (Jupter Notebook) -> Comparar os Erros, qual modelo esta errando menos;

    # future = m.make_future_dataframe(periods=5)
    # future.tail()

    forecast = m.predict(train_set)
    # Validar documentacao do m.predict -> devemos passar train_data_set ?;
    # Recomendacao final - Processo de inicio meio e fim -> Validar ligação do ROE e metricas para recomendar as ações a longo prazo;


    # forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    print(forecast)

    fig1 = m.plot(forecast)

    fig2 = m.plot_components(forecast)

    plot_plotly(m, forecast)
    # plot_components_plotly(m, forecast)

    plt.show()

