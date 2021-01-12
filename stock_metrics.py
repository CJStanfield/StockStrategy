import pandas as pd
import numpy as np

SMA = '{}_SMA_{}'
EMA = '{}_EMA_{}'
INCREASE_DECREASE = '{}_increase_decrease'
DERIVATIVE = '{}_derivative'
DISTANCE = '{}_{}_distance'
RSI = 'rsi'
OBV = 'obv'
MACD = 'macd'
MACD_SIGNAL = 'macd_signal'
MACD_DECISION = 'macd_decision'
CLOSE_PRICE = 'Close'
VOLUME = 'Volume'


def increase_decrease(data, col):
    data['temp'] = data[col].shift(-1)
    data[INCREASE_DECREASE.format(col)] = data.apply(lambda x: 1 if x['temp'] - x[col] > 0 else 0, axis=1)
    data = data.drop(columns=['temp'])
    return data


def column_derivative_metric(col, data):
    data[DERIVATIVE.format(col)] = data[col].diff()
    return data


def distance_metric(data, col1, col2):
    data[DISTANCE.format(col1, col2)] = (data[col1] - data[col2]) / data[col2]
    return data


def rsi_metric(data, col, window=14):
    # 1: Compute price movement each period
    # 2: Compute average gain/loss over last 14 days
    gain = pd.DataFrame(data[col].rolling(2).apply(lambda x: x.iloc[1] - x.iloc[0]))
    gain[gain < 0] = np.nan
    gain = gain.rolling(window=window, min_periods=1).mean()
    gain = gain.fillna(0)

    loss = pd.DataFrame(data[col].rolling(2).apply(lambda x: x.iloc[1] - x.iloc[0]))
    loss[loss > 0] = np.nan
    loss = loss.abs()
    loss = loss.rolling(window=window, center=True, min_periods=1).mean()
    loss = loss.fillna(0)
    # 3: Calculate RS and RSI
    relative_strength = gain / loss
    relative_strength_index = 100 - 100 / (1 + relative_strength)
    data[RSI] = relative_strength_index
    return data


def macd_metric(data):
    # 9 period exponential moving average
    macd_signal = data[CLOSE_PRICE].ewm(span=9, adjust=False).mean()
    # 12 period exponential moving average
    EMA_12_day = data[CLOSE_PRICE].ewm(span=12, adjust=False).mean()
    # 26 period exponential moving average
    EMA_26_day = data[CLOSE_PRICE].ewm(span=26, adjust=False).mean()
    macd = EMA_26_day - EMA_12_day
    data[MACD] = macd
    data[MACD_SIGNAL] = macd_signal
    data[MACD_DECISION] = macd - macd_signal


def on_balance_volume(data, span=12):
    # OBV = Previous OBV + Current Trading Volume
    data[OBV] = np.where(data[CLOSE_PRICE] > data[CLOSE_PRICE].shift(1), data[VOLUME],
                              np.where(data[CLOSE_PRICE] < data[CLOSE_PRICE].shift(1),
                                       -data[VOLUME], 0)).cumsum()
    data = exponential_moving_average(data, OBV, span=span)
    data = column_derivative_metric(EMA.format(OBV, span), data)
    return data


def simple_moving_average(data, col, window):
    data[SMA.format(col, window)] = data[col].rolling(window=window).mean()
    return data


def exponential_moving_average(data, col, span):
    data[EMA.format(col, span)] = data[col].ewm(span=span, adjust=False).mean()
    return data
