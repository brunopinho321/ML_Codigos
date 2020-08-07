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

    def MSE_(self, y_true, y_predict):
        mse = []
        for i in y_predict:
            mse.append(self.MSE(y_true, i))
        return mse
class Gradiente_Regressao_Linear_Simples():
    def __init__(self, epocas, taxa_aprendizado):
        self.epocas = epocas
        self.alfa = taxa_aprendizado
    def calcular_G0(self):
        y_pre  = (self.w_1 * self.X)
        self.e = self.y - (y_pre + self.w_0)
        return self.alfa * np.sum(self.e)/len(self.e)
    def calcular_G1(self):
        self.e = self.y - (np.dot(self.w_1, self.X) + self.w_0)
        return self.alfa * np.sum(np.dot(self.e, self.X))/len(self.e)
    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.w_0 = 0#np.random.normal(0,5)
        self.w_1 = 0#np.random.normal(0,5)
        
        for i in range(self.epocas):
            self.w_0 = self.w_0 + self.calcular_G0()
            self.w_1 = self.w_1 + self.calcular_G1()
        self.w = np.c_[self.w_0, self.w_1]
    def predict(self, X):
        return np.dot(self.w_1, np.array(X)) + self.w_0


class Gradiente_Regressao_Linear_Mutipla():
    def __init__(self, epocas , taxa_aprendizado):
        self.epocas = epocas
        self.alfa = taxa_aprendizado
    def calcular_G(self):
        y_predict = self.X_ @ self.w
        erro = self.y - y_predict
        gradiente = np.sum(erro @ self.X_)/len(self.X_)
        return gradiente
    def fit(self, X, y):
        n = X.shape[0]
        self.y_predic_epocas = []
        self.y = y
        self.X_ = np.c_[np.ones(n), X]
        self.w = np.zeros(len(self.X_[0]))
        self.epocas_ = []
        for i in range(self.epocas):
            self.w += self.alfa * self.calcular_G() 
        
            y_predict = self.X_ @ self.w
            self.y_predic_epocas.append(y_predict)
            self.epocas_.append(i+1)
    def predict(self, X):
        n = X.shape[0]
        X_ = np.c_[np.ones(n),X]
        return((X_ @ self.w))


class Gradiente_Regressao_Linear_Mutiplar_Reguralizada():
    def __init__(self, epocas , taxa_aprendizado, termo_regularizacao):
        self.epocas = epocas
        self.alfa = taxa_aprendizado
        self.termo_regularizacao = termo_regularizacao
    def calcular_G(self):
        y_predict = self.X_ @ self.w
        erro = self.y - y_predict
        gradiente = (np.sum((erro @ self.X_) - ((self.termo_regularizacao / len(self.X_)) * self.w))) / len(self.X_)
        return gradiente
    def fit(self, X, y):
        self.y = y
        n = X.shape[0] 
        self.X_ = np.c_[np.ones(n), X]
        self.w = np.zeros(len(self.X_[0]))
       
        for i in range(self.epocas):
            self.w[0] += np.sum(self.y - (self.X_ @ self.w))/len(self.y)
            self.w[1:] += self.alfa * self.calcular_G() 
    def predict(self, X):
        n = X.shape[0]
        X_ = np.c_[np.ones(n),X]
        return((X_ @ self.w))

class Gradiente_Estocastico_Regressao_Linear_Mutipla():
    def __init__(self, epocas , taxa_aprendizado):
        self.epocas = epocas
        self.alfa = taxa_aprendizado
        self.y_predic_epocas = []
    def fit(self, X, y):
        n = X.shape[0]
        self.X_ = np.c_[np.ones(n), X]
        self.w = np.zeros(len(self.X_[0]))
        y_predict2 = []
        self.epocas_ = []
        for i in range(self.epocas):
            for j in range (len(self.X_)):
                y_predict = self.X_[j] @ self.w.T
                erro = y[j] - y_predict
                gradiente = erro * self.X_[j]
                self.w += self.alfa * gradiente
            y_predict2 = self.X_ @ self.w
            self.y_predic_epocas.append(y_predict2)
            self.epocas_.append(i+1)
    def predict(self, X):
        n = X.shape[0]
        X_ = np.c_[np.ones(n),X]
        return((X_ @ self.w))

if __name__ == "__main__":
    #data = np.loadtxt("C:\\Users\\Bruno\\Desktop\\Nova pasta\\advertising.csv", skiprows=1, delimiter=",")
    data = np.loadtxt("./housing.data")
    #np.random.shuffle(data)
    X = data[:, -2]
    y = data[:, -1]

    n = X.shape[0]
    n_train = int(n*0.8)
    n_test = n - n_train
    X_train = X[:n_train]
    X_test = X[-n_test:]
    y_train = y[:n_train]
    y_test = y[-n_test:]

    print(y)
    

    g = Gradiente_Estocastico_Regressao_Linear_Mutipla(1000, 0.00001)
    g.fit(X_train, y_train)
    m = Metricas()
    print(m.MSE_(y_train, g.y_predic_epocas))

    g = Gradiente_Regressao_Linear_Mutipla(1000, 0.00001)
    g.fit(X_train, y_train)
    print(m.MSE_(y_train, g.y_predic_epocas))

    













