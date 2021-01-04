import pandas as pd
import math


class Strategy:
    SELL = 0
    BUY = 1
    HOLD = 2
    TREND = -1

    metrics = ['SMA_5', 'SMA_20']
    cash = 1000
    shares = 0

    def __init__(self, data):
        self.data = data

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
