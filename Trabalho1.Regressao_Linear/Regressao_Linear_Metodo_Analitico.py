import numpy as np

class Metricas():
    def __init__(self):
        pass
    def RSS(self, y_true, y_predict):
        return np.sum(((y_true - y_predict)**2))

    def RSE(self, y_true, y_predict):
        return np.sqrt(((1/(len(y_true)-2)) * self.RSS(y_true, y_predict)))

    def R2(self, y_true, y_predict):
        tss = np.sum(((y_true - np.mean(y_true))**2))
        return 1 - (self.RSS(y_true, y_predict)/tss)

    def MAE(self, y_true, y_predict):
        n = len(y_true)
        return (np.sum(abs(y_true - y_predict)))/n
    def MSE(self, y_true, y_predict):
        return (1/len(y_true)) * np.sum((y_true - y_predict)**2)

class Regressao_Linear_Simples():
    def __init__(self): 
        pass

    def fit(self, X, y):
        y_bar = np.mean(y)
        x_bar = np.mean(X)

        x_x_bar = X - x_bar
        y_y_bar = y - y_bar
        
        num = np.sum(x_x_bar * y_y_bar)

        denom = np.sum(x_x_bar * x_x_bar)

        self.b1 = num/denom
        self.b0 = y_bar - self.b1 * x_bar
        self.w = np.c_[self.b0, self.b1]
    def predict(self, X):
        return self.b0 + X*self.b1


class Regressao_Linear_Mutipla():
    def __init__(self):
        pass

    def fit(self, X, y):
        n = X.shape[0]
        X_ = np.c_[np.ones(n), X]

        self.b = np.linalg.pinv(X_.T @ X_) @ X_.T @ y
        self.w = self.b
    def predict(self, X):
        n = X.shape[0]
        X_ = np.c_[np.ones(n), X]

        return X_ @ self.b

class Regressao_Quadratica():
    def __init__(self):
        pass
    def fit(self, X, y):
        X_ = np.array(X*X)
        X2 = np.c_[X, X_]
        self.rlm = Regressao_Linear_Mutipla()
        self.rlm.fit(X2, y)
        self.w = self.rlm.b
    def predict(self, X):
        X_ = np.array(X*X)
        X2 = np.c_[X, X_]
        return self.rlm.predict(X2)
        
class Regressao_Cubica():
    def __init__(self):
        pass
    def fit(self, X, y):
        X3 = np.c_[X, X * X, X * X * X]
        self.rlm = Regressao_Linear_Mutipla()
        self.rlm.fit(X3, y)
        self.w = self.rlm.b
    def predict(self, X):
        X3 = np.c_[X, X*X, X*X*X]
        return self.rlm.predict(X3)
if __name__ == "__main__":
    data = np.loadtxt("./housing.data")
    np.random.shuffle(data)
    X = data[:, -2]
    y = data[:, -1]

    n = X.shape[0]
    n_train = int(n*0.8)
    n_test = n - n_train
    X_train = X[:n_train]
    X_test = X[-n_test:]
    y_train = y[:n_train]
    y_test = y[-n_test:]

    gr = Regressao_Linear_Simples()
    gr.fit(X_train, y_train)
    
    data = np.loadtxt("./trab1_data.txt")
    np.random.shuffle(data)
    y = data[: ,-1]
    X = data.T[: 5].T
    n = X.shape[0]
    n_train = int(n*0.8)
    n_test = n - n_train
    X_train = X[:n_train]
    X_test = X[-n_test:]
    y_train = y[:n_train]
    y_test = y[-n_test:]

    r =Regressao_Linear_Mutipla()
    r.fit(X_train, y_train)
    y_p = r.predict(X_test)

    print(y_p)