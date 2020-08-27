import numpy as np
import math


class RegressaoLogistica:
    def __init__(self, alfa = 0.00000000000000000000001958, epocas=6000):
        self.alfa = alfa
        self.epocas = epocas
        self.ws = None
        self.bias = None
    def fit(self, X, y):
        n, m = X.shape
        self.N = n
        self.ws = np.zeros(m)
        self.bias = 0

        for i in range(self.epocas):
            modeloLinear = X @ self.ws + self.bias
            y_pred = self.logistica(modeloLinear)

            dw = (1/n) * (X.T @ (y - y_pred ))
            db = (1/n) * sum(y - y_pred)

            self.ws += self.alfa * dw
            self.bias += self.alfa * db

    def predict(self, X):
        modeloLinear = (X @ self.ws) + self.bias
        y_pred = self.logistica(modeloLinear)
        y_predicted = []
        for i in y_pred:
            if i > 0.5:
                y_predicted.append(1.0)
            else:
                y_predicted.append(0.0)
        return np.array(y_predicted)
    def logistica(self, x):
        return 1/ (1 + np.exp(-x))
if __name__ == "__main__":
    data = np.loadtxt(".\ex2data1.txt", skiprows=1, delimiter=",")
    np.random.shuffle(data)
    X = data[:, 0: -1]
    y = data[: , 2]
    n = X.shape[0]
    n_train = int(n*0.7)
    n_test = n - n_train
    X_train = X[:n_train]
    X_test = X[-n_test:]
    y_train = y[:n_train]
    y_test = y[-n_test:]
   