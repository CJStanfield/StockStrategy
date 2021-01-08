import pandas as pd
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt


class Strategy:
    SELL = 0
    BUY = 1
    HOLD = 2
    TREND = -1

    metrics = ['SMA_5', 'SMA_20']
    cash = 1000
    shares = 0

    def __init__(self, data):
        self.data = pd.DataFrame(data)

    def get_data(self):
        return self.data

    def execute_trade(self, decision, shares, price):
        if decision == self.BUY:
            self.cash = self.cash - (shares * price)
            self.shares = self.shares + shares
            print("Trade Executed. \tShares: {}\tPrice: {}\tTrade Type: {}\tTotal Value: {}".format(
                shares, price, "BUY", (self.cash + self.shares * price)
            ))
        if decision == self.SELL:
            self.cash = self.cash + (shares * price)
            self.shares = self.shares - shares
            print("Trade Executed. \tShares: {}\tPrice: {}\tTrade Type: {}\tTotal Value: {}".format(
                shares, price, "SELL", (self.cash + self.shares * price)
            ))
        pass

    def decision(self, row):
        if self.cash <= 0:
            print("Out of Cash!!!")
            return
        if row['SMA_5'] >= row['SMA_20'] and self.TREND == 0:
            self.TREND = 1
            shares = math.floor(self.cash / row['Close'])
            self.execute_trade(self.BUY, shares, row['Close'])
        if row['SMA_5'] <= row['SMA_20'] and self.TREND == 1:
            self.TREND = 0
            self.execute_trade(self.SELL, self.shares, row['Close'])
        if row['SMA_5'] >= row['SMA_20'] and self.TREND == -1:
            self.TREND = 1
            shares = math.floor(self.cash / row['Close'])
            self.execute_trade(self.BUY, shares, row['Close'])

    def backtest(self):
        self.data.apply(lambda row: self.decision(row), axis=1)
        current_price = self.data['Close'][self.data.last_valid_index()]
        print("ACCOUNT\tShares: {}\tCash: {}\tTotal Value: {}".format(
            self.shares, self.cash, (self.cash + self.shares * current_price)
        ))
        return

    def generate_metrics(self):
        self.rsi_metric()
        self.macd_metric()
        self.on_balance_volume()
        self.simple_moving_average(5)
        self.simple_moving_average(20)
        self.exponential_moving_average(5)
        self.exponential_moving_average(20)
        self.data['SMA Distance'] = self.data['SMA_5'] - self.data['SMA_20']
        self.data['EMA Distance'] = self.data['EMA_5'] - self.data['EMA_20']
        self.data['Increase/Decrease'] = self.data['Close'].rolling(2).apply(lambda x: 1 if x.iloc[1] - x.iloc[0] > 0 else 0)
        return

    def rsi_metric(self, window=14):
        # 1: Compute price movement each period
        # 2: Compute average gain/loss over last 14 days
        gain = pd.DataFrame(self.data['Close'].rolling(2).apply(lambda x: x.iloc[1] - x.iloc[0]))
        gain[gain < 0] = np.nan
        gain = gain.rolling(window=window, min_periods=1).mean()
        gain = gain.fillna(0)

        loss = pd.DataFrame(self.data['Close'].rolling(2).apply(lambda x: x.iloc[1] - x.iloc[0]))
        loss[loss > 0] = np.nan
        loss = loss.abs()
        loss = loss.rolling(window=window, center=True, min_periods=1).mean()
        loss = loss.fillna(0)
        # 3: Calculate RS and RSI
        relative_strength = gain / loss
        relative_strength_index = 100 - 100 / (1 + relative_strength)
        self.data['RSI'] = relative_strength_index

    def macd_metric(self):
        # 9 period exponential moving average
        macd_signal = self.data['Close'].ewm(span=9, adjust=False).mean()
        # 12 period exponential moving average
        EMA_12_day = self.data['Close'].ewm(span=12, adjust=False).mean()
        # 26 period exponential moving average
        EMA_26_day = self.data['Close'].ewm(span=26, adjust=False).mean()
        macd = EMA_26_day - EMA_12_day
        self.data['MACD'] = macd
        self.data['MACD_Signal'] = macd_signal
        self.data['MACD_Decision'] = macd - macd_signal

    def on_balance_volume(self, span=12):
        # OBV = Previous OBV + Current Trading Volume
        self.data['On Balance Volume'] = np.where(self.data['Close'] > self.data['Close'].shift(1), self.data['Volume'], np.where(self.data['Close'] < self.data['Close'].shift(1), -self.data['Volume'], 0)).cumsum()
        self.data['On Balance Volume Smooth'] = self.data['On Balance Volume'].ewm(span=span, adjust=False).mean()
        self.data['On Balance Volume Derivative'] = self.data['On Balance Volume Smooth'].diff()

    def simple_moving_average(self, window):
        self.data['SMA_{}'.format(window)] = self.data['Close'].rolling(window=window).mean()

    def exponential_moving_average(self, span):
        self.data['EMA_{}'.format(span)] = self.data['Close'].ewm(span=span, adjust=False).mean()

    def generate_neural_net(self):
        training_data = self.data.dropna()
        X = training_data[['RSI',
                           'On Balance Volume Derivative',
                           'MACD_Decision',
                           'SMA Distance',
                           'EMA Distance']]
        x = X.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform((x))
        X = pd.DataFrame(x_scaled)
        y = training_data['Increase/Decrease']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        model = Sequential()
        model.add(Dense(12, input_dim=5, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # fit the keras model on the dataset
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=10)
        self.graph_neural_network_results(history)
        return

    def graph_neural_network_results(self, history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

