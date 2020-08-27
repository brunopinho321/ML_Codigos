import numpy as np
import math
from sklearn.linear_model import LogisticRegression
ALPHA = 0.00000001
NUMB_OF_ITERATIONS =5000






class LogisticRegression1:
    def __init__(self):
        pass
    def fit(self, X, y):
        m = X.shape[1]
        n = X.shape[0]
        X_ = np.c_[np.zeros(n), X]
        self.w = np.zeros((np.shape(X_)[1],1))
        #print(self.w)
        for i in range(NUMB_OF_ITERATIONS):
            y_pred = self.logistic(X_ @ self.w)
            erro = y - y_pred
            #(1 / n) * np.dot(X.T, (y - y_predicted))
            gradiente = (1/n) * np.sum(erro @ X_)

            #self.w += ALPHA * gradiente
            self.w = self.w + ALPHA * gradiente
    def predict(self, X):
        m = X.shape[1]
        n = X.shape[0]
        X_ = np.c_[np.ones(n), X]
        #print(self.w)
        modelo = X_ @ self.w
        y_pred = self.logistic((modelo))
        y_mean = np.mean(y_pred)
        print(np.median(y_pred))
        y_pred_cls = [1.0 if i > np.median(y_pred) else 0.0 for i in y_pred]
        return np.array(y_pred_cls)

        

    def logistic(self, X):
        return  1 / (1 + np.exp(-X))
    





class LogisticRegression2:
    def __init__(self):
        pass
    def separate_by_classes(self, X, y):
        self.classes = np.unique(y)
        classes_index = { }
        subdatasets = { }
        X_ = np.c_[X, y]
        cls, counts = np.unique(y, return_counts=True)
        #frequencia que cada classe ocorre
        self.class_freq = dict(zip(cls, counts))
        for class_type  in self.classes:
            classes_index[class_type] = np.argwhere(y == class_type)
            subdatasets[class_type] = X[classes_index[class_type], :]
            #self.class_freq[class_type] = self.class_freq[class_type]/sum(list(self.class_freq.values()))
        dados = {}
        dados2 = []
        for class_type in self.classes:
            for i in range(len(y)):
                if y[i] == class_type:
                    dados2.append(X[i])
            dados[class_type] = np.array(dados2)
            dados2=[]
        return dados


    def gradiente(self, X, y):
        m = X.shape[1]
        n = X.shape[0]
        w = np.zeros(m)
        #w = np.random.normal(0,1, size = X[0].shape)
        for i in range(NUMB_OF_ITERATIONS):
            y_pred = 1 / (1 + np.exp(-(X @ w)))
            erro = y - y_pred
            gradiente = np.sum(erro @ X)/n
            w += ALPHA * gradiente
        return w


    def fit(self, X, y):
        n = X.shape[0]
        X_ = np.c_[np.ones(n), X]
    
        self.X = self.separate_by_classes(X_, y)
        self. prob = []
        self.w_classes = {}
        for c in self.classes:
            y_train = [c] * self.class_freq[c]
            self.w_classes[c] = self.gradiente(self.X[c], y_train)
    def prob(self, x, w):
        return 1.0/ (1 + np.exp(-(x @ w)))
        
    def predict_prob(self, X):
        max_prob = 0
        #pred_class = None
        #print(1/(1 + X @ self.w_classes[0]))
        
        for c in self.classes:
            p = 1/(1 + np.exp(-(X @ self.w_classes[c])))
            print(str(c) + " : "+ str(p))
            if(max_prob < p):
                max_prob = p
                pred_class = c
        return pred_class
    def predict(self, X):
        n = X.shape[0]
        X_ = np.c_[np.ones(n), X]
        pred = []
        for x in X_:
           pred.append(self.predict_prob(x))
        return np.array(pred)
    def logistic(self, ws):
        return 1.0 / (1 + np.exp(-ws))






class LogisticRegression3:

    def __init__(self, alfa=0.000001, epocas=1000):
        self.alfa = alfa
        self.epocas = epocas
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n, m = X.shape
        
        self.w = np.zeros(m)
        self.bias = 0

        for i in range(self.epocas):
            
            modelo = np.dot(X, self.w) + self.bias
            
            y_pred= self.logistic(modelo)

            dw = (1 / n) * (X.T @ (y - y_pred))
            db = (1 / n) * np.sum(y - y_pred)
  
            self.w += self.alfa * dw
            self.bias += self.alfa * db
        
        
    def predict(self, X):
        modelo = (X @ self.w) + self.bias
        y_pred = self.logistic(modelo)
        y_mean = np.mean(y_pred)
        #print(y_mean)
        y_cls = [1.0 if i > np.median(y_pred) else 0.0 for i in y_pred]
        return np.array(y_cls)

    def logistic(self, x):
        return 1 / (1 + np.exp(-x))

class RegressaoLogistica:
    def __init__(self, alfa = 0.00000000000000000000001958, epocas=6000):
        self.alfa = alfa
        self.epocas = epocas
        self.pesos = None
        self.bias = None
    def fit(self, X, y):
        n, m = X.shape
        self.N = n
        self.pesos = np.zeros(m)
        self.bias = 0

        for i in range(self.epocas):
            modeloLinear = X @ self.pesos + self.bias
            y_pred = self.logistica(modeloLinear)

            dw = (1/n) * (X.T @ (y - y_pred ))
            db = (1/n) * sum(y - y_pred)

            self.pesos += self.alfa * dw
            self.bias += self.alfa * db

    def predict(self, X):
        modeloLinear = (X @ self.pesos) + self.bias
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
    g = LogisticRegression3()
    g.fit(X_train, y_train)
    a = g.predict(X_test)
   