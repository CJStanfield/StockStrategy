import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import strategy as Strategy

START_DATE = '2020-12-01'
END_DATE = '2020-12-31'
STOCK = 'aapl'

def get_stock_data(symbol):
    stock_df = yf.download(symbol,
                           start=START_DATE,
                           END_DATE=END_DATE,
                           interval='30m',
                           progress=False)
    return stock_df

def plot(data, symbol):
    # data['Close'].plot(title="{} stock price".format(symbol))
    data['On Balance Volume Derivative'].plot()
    # data['SMA_20'].plot()
    # data['On Balance Volume'].plot()
    # data['On Balance Volume Smooth'].plot()
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = get_stock_data(STOCK)
    # data = simple_moving_average(data, window=5)
    # data = simple_moving_average(data, window=20)
    # data = data.dropna()
    strategy = Strategy.Strategy(data)
    strategy.generate_metrics()
    strategy.generate_neural_net()
    # plot(data, STOCK)

