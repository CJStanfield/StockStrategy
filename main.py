import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import strategy as Strategy

START_DATE = '2019-01-01'
END_DATE = '2020-12-31'
STOCK = 'fb'

def get_stock_data(symbol):
    stock_df = yf.download(symbol,
                           start=START_DATE,
                           END_DATE=END_DATE,
                           progress=False)
    return stock_df

def simple_moving_average(data, window):
    data['SMA_{}'.format(window)] = data['Close'].rolling(window=window).mean()
    return data

def plot(data, symbol):
    data['Close'].plot(title="{} stock price".format(symbol))
    data['SMA_5'].plot()
    data['SMA_20'].plot()
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = get_stock_data(STOCK)
    data = simple_moving_average(data, window=5)
    data = simple_moving_average(data, window=20)
    data = data.dropna()

    strat = Strategy.Strategy(data)
    strat.backtest()
    plot(data, STOCK)
    x = 5

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
