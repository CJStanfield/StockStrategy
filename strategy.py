import pandas as pd
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


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
        # EMA 50 will be the trend line used in the bayes calculation
        self.exponential_moving_average(50)

        # self.data['SMA Distance'] = self.data['SMA_5'] - self.data['SMA_20']
        # self.data['EMA Distance'] = self.data['EMA_5'] - self.data['EMA_20']
        self.data['temp'] = self.data['Close'].shift(-1)
        self.data['Increase/Decrease'] = self.data.apply(lambda x: 1 if x['temp'] - x['Close'] > 0 else 0, axis=1)
        self.data = self.data.drop(columns=['temp'])
        return

        # self.data['SMA Distance'] = self.data.apply(lambda x: 1 if x['SMA_5'] - x['SMA_20'] > 0 else 0, axis=1)
        # self.data['EMA Distance'] = self.data.apply(lambda x: 1 if x['EMA_5'] - x['EMA_20'] > 0 else 0, axis=1)

    def linear_regression_model(self):
        temp_data = self.data[['Close', 'EMA_50']]
        temp_data['EMA_50_Distance'] = (temp_data['Close'] - temp_data['EMA_50'])/temp_data['EMA_50']
        temp_data['EMA_50_Slope'] = temp_data['EMA_50'].diff()
        temp_data['EMA_50_Crossover_Count'] = np.nan
        i = 1
        trend = 1
        for index, row in temp_data.iterrows():
            if row['EMA_50_Slope'] > 0:
                if trend == 1:
                    i = i + 1
                else:
                    trend = 1
                    i = 1
            elif row['EMA_50_Slope'] < 0:
                if trend == 0:
                    row['EMA_50_Crossover_Count'] = i
                    i = i + 1
                else:
                    trend = 0
                    i = 1
            temp_data.at[index, 'EMA_50_Crossover_Count'] = i
        temp_data['temp'] = temp_data['EMA_50_Slope'].shift(-1)
        temp_data['EMA_Increase_Decrease'] = temp_data.apply(lambda x: 1 if x['temp'] > 0 else 0, axis=1)
        temp_data = temp_data.dropna()
        temp_data = temp_data.drop(columns=['temp'])
        X = temp_data[['EMA_50_Distance', 'EMA_50_Slope', 'EMA_50_Crossover_Count']]
        y = temp_data['EMA_Increase_Decrease']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        lr_model = LinearRegression().fit(X_train, y_train)
        r_sq = lr_model.score(X_train, y_train)

        predictions = lr_model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)


        #Compute rolling window base probability
        #Identify positive slope in EMA line from an inflection point
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

    def generate_neural_net(self, indicators, lookback=60, epochs=50):
        training_data = self.data.dropna()
        min_max_scaler = preprocessing.MinMaxScaler()

        # x_scaled = min_max_scaler.fit_transform(training_data[indicators].values)
        # X = np.array(x_scaled)
        X = np.array(training_data[indicators].values)
        y = np.array(training_data['EMA_Increase_Decrease'])

        # Set up lookback data for LSTM. Currently set to 60 days
        temp_X = []
        for i in range(lookback, X.shape[0]+1):
            temp_X.append(X[i - lookback:i, :])

        # Reshape data for LSTM
        X = np.array(temp_X)
        # X = np.reshape(X, (X.shape[0], X.shape[1], 5))
        y = y[lookback-1:]

        # Training/Testing Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        # Create a model with keras
        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(64, input_shape=(lookback, X[0].shape[1]), return_sequences=True, activation='relu'))
        model.add(Dropout(0.2))
        model.add(LSTM(12, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.summary()

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        # fit the keras model on the dataset
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=10)
        return history

    def graph_neural_network_results(self, history):
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
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

