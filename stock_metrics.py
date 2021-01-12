import pandas as pd
import numpy as np

class metrics:
    SMA = '{}_sma_{}'
    EMA = '{}_ema_{}'
    INCREASE_DECREASE = '{}_increase_decrease'
    CROSSOVER_COUNT = '{}_crossover_count'
    DERIVATIVE = '{}_derivative'
    DISTANCE = '{}_{}_distance'
    CLASSIFICATION = '{}_classification'
    RSI = 'rsi'
    OBV = 'obv'
    MACD = 'macd'
    MACD_SIGNAL = 'macd_signal'
    MACD_DECISION = 'macd_decision'
    CLOSE_PRICE = 'Close'
    VOLUME = 'Volume'


    def __init__(self, close, volume):
        self.VOLUME = volume
        self.CLOSE_PRICE = close

    def increase_decrease(self, data, col):
        data['temp'] = data[col].shift(-1)
        data[self.INCREASE_DECREASE.format(col)] = data.apply(lambda x: 1 if x['temp'] - x[col] > 0 else 0, axis=1)
        data = data.drop(columns=['temp'])
        return data


    def column_derivative_metric(self, col, data):
        data[self.DERIVATIVE.format(col)] = data[col].diff()
        return data


    def distance_metric(self, data, col1, col2):
        data[self.DISTANCE.format(col1, col2)] = (data[col1] - data[col2]) / data[col2]
        return data


    def rsi_metric(self, data, window=14):
        # 1: Compute price movement each period
        # 2: Compute average gain/loss over last 14 days
        gain = pd.DataFrame(data[self.CLOSE_PRICE].rolling(2).apply(lambda x: x.iloc[1] - x.iloc[0]))
        gain[gain < 0] = np.nan
        gain = gain.rolling(window=window, min_periods=1).mean()
        gain = gain.fillna(0)

        loss = pd.DataFrame(data[self.CLOSE_PRICE].rolling(2).apply(lambda x: x.iloc[1] - x.iloc[0]))
        loss[loss > 0] = np.nan
        loss = loss.abs()
        loss = loss.rolling(window=window, center=True, min_periods=1).mean()
        loss = loss.fillna(0)
        # 3: Calculate RS and RSI
        relative_strength = gain / loss
        relative_strength_index = 100 - 100 / (1 + relative_strength)
        data[self.RSI] = relative_strength_index
        return data


    def macd_metric(self, data):
        # 9 period exponential moving average
        macd_signal = data[self.CLOSE_PRICE].ewm(span=9, adjust=False).mean()
        # 12 period exponential moving average
        EMA_12_day = data[self.CLOSE_PRICE].ewm(span=12, adjust=False).mean()
        # 26 period exponential moving average
        EMA_26_day = data[self.CLOSE_PRICE].ewm(span=26, adjust=False).mean()
        macd = EMA_26_day - EMA_12_day
        data[self.MACD] = macd
        data[self.MACD_SIGNAL] = macd_signal
        data[self.MACD_DECISION] = macd - macd_signal
        return data


    def on_balance_volume(self, data, span=12):
        # OBV = Previous OBV + Current Trading Volume
        data[self.OBV] = np.where(data[self.CLOSE_PRICE] > data[self.CLOSE_PRICE].shift(1), data[self.VOLUME],
                                  np.where(data[self.CLOSE_PRICE] < data[self.CLOSE_PRICE].shift(1),
                                           -data[self.VOLUME], 0)).cumsum()
        data = self.exponential_moving_average(data, self.OBV, span=span)
        data = self.column_derivative_metric(self.EMA.format(self.OBV, span), data)
        return data


    def simple_moving_average(self, data, col, window):
        data[self.SMA.format(col, window)] = data[col].rolling(window=window).mean()
        return data


    def exponential_moving_average(self, data, col, span):
        data[self.EMA.format(col, span)] = data[col].ewm(span=span, adjust=False).mean()
        return data

    def ema_trend_indicator(self, data, ema_span):
        # Required Column Names
        col_ema_name = self.EMA.format(self.CLOSE_PRICE, ema_span)
        col_derivative = self.DERIVATIVE.format(col_ema_name)
        col_distance = self.DISTANCE.format(col_ema_name, col_derivative)
        col_crossover_count = self.CROSSOVER_COUNT.format(col_ema_name)
        col_ema_slope_classification = self.CLASSIFICATION.format(col_ema_name)

        data[col_distance] = (data[self.CLOSE_PRICE] - data[col_ema_name]) / data[col_ema_name]
        data[col_derivative] = data[col_ema_name].diff()
        data[col_crossover_count] = np.nan

        i = 1
        trend = 1
        for index, row in data.iterrows():
            if row[col_derivative] > 0:
                if trend == 1:
                    i = i + 1
                else:
                    trend = 1
                    i = 1
            elif row[col_derivative] < 0:
                if trend == 0:
                    row[col_crossover_count] = i
                    i = i + 1
                else:
                    trend = 0
                    i = 1
            data.at[index, col_crossover_count] = i
        data['temp'] = data[col_derivative].shift(-1)
        data[col_ema_slope_classification] = data.apply(lambda x: 1 if x['temp'] > 0 else 0, axis=1)
        data = data.dropna()
        data = data.drop(columns=['temp'])

        return data
