import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
START_DATE = '2019-01-01'
END_DATE = '2020-12-31'


def get_stock_data(symbol):
    stock_df = yf.download(symbol,
                             start=START_DATE,
                             END_DATE=END_DATE,
                             progress=False)
    stock_df['Close'].plot(title="{} stock price".format(symbol))
    plt.show()
    pass

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    get_stock_data('aapl')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
