import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import strategy as Strategy

START_DATE = '2005-01-01'
END_DATE = '2020-12-31'
STOCK = 'aapl'

def get_stock_data(symbol):
    stock_df = yf.download(symbol,
                           start=START_DATE,
                           END_DATE=END_DATE,
                           progress=False)
    return stock_df

def plot(data, symbol):
    # data['SMA_5'].plot()
    # data['SMA_20'].plot()
    # data['EMA_5'].plot()
    # data['EMA_20'].plot()
    # data['Close'].plot(title="{} stock price".format(symbol))
    plt.show()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    indicators = ['RSI',
                   'On Balance Volume Derivative',
                   'MACD_Decision',
                   'SMA Distance',
                   'EMA Distance',
                  'EMA_Increase_Decrease']
    indicators2 = ['MACD_Decision']
    data = get_stock_data(STOCK)
    strategy = Strategy.Strategy(data, close='Close', volume='Volume')
    strategy.generate_metrics()
    strategy.linear_regression_model()
    # history = strategy.generate_neural_net(indicators=indicators2, lookback=30)
    # strategy.graph_neural_network_results(history)


