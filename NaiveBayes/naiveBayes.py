import numpy as np
from sklearn.naive_bayes import GaussianNB

class NaiveBayes:
    def __init__(self):
        pass
    def variancia(self, X):
        X = X- X.mean(axis = 0)
        return (sum((X**2))/(len(X)))
    
    def prob(self, indice_classe, x):
        media = self.medias[indice_classe]
        var = self.variancias[indice_classe]
        exp = np.exp(- (x - media)**2 / (2 * var))
        return exp/ np.sqrt(2 * np.pi * var)
    
    def fit(self, X, y):
        n, m = X.shape
        self.classes = np.unique(y)
        n_c = len(self.classes)


        self.medias = np.zeros((n_c, m), dtype=np.float64)
        self.variancias = np.zeros((n_c, m), dtype=np.float64)
        self.p_anteriores = np.zeros(n_c, dtype=np.float64)

        for i, c in enumerate(self.classes):
            X_ = X[c==y]
            self.medias[i,:] = X_.mean(axis=0)
            self.variancias[i,:] = self.variancia(X_)
            self.p_anteriores[i] = X_.shape[0] / float(n)
        print(self.variancias)
    
    def predict_prob(self, X):
        p_posteriores = []

        for i, c in enumerate(self.classes):
            anterior = np.log(self.p_anteriores[i])
            posterior = np.sum(np.log(self.prob(i, X)))
            posterior += anterior
            p_posteriores.append(posterior)
        return self.classes[np.argmax(p_posteriores)]     

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.predict_prob(x))
        return np.array(y_pred)
    
    

data = np.loadtxt("c:/Users/bruno/Desktop/teste/ex2data1.txt", skiprows=1, delimiter=",")
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


g = NaiveBayes()
g.fit(X_train, y_train)
print(g.predict(X_test))
nb =GaussianNB()
nb.fit(X_train, y_train)
print(nb.predict(X_test))