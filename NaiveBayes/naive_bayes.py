import numpy as np
import math 
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from teste import gaussClf
class NaiveBayesGaussiano():
    def __init__(self):
        self.medias = {}
        self.variancias = {}
    def separar_classes(self, X, y):
        X2=X
        self.classes = np.unique(y)
        classes_index = {}
        subdatasets = {}
        X_ = np.c_[X, y]
        cls, counts = np.unique(y, return_counts=True)
        #frequencia que cada classe ocorre
        self.class_freq = dict(zip(cls, counts))
        for class_type  in self.classes:
            classes_index[class_type] = np.argwhere(y == class_type)
            subdatasets[class_type] = X2[classes_index[class_type], :]
            self.class_freq[class_type] = self.class_freq[class_type]/sum(list(self.class_freq.values()))
        dados = {}
        dados2 = []
        for class_type in self.classes:
            for i in range(len(y)):
                if y[i] == class_type:
                    dados2.append(X[i])
            dados[class_type] = np.array(dados2)
            dados2=[]
        return dados
    
    
    def fit(self, X, y):
        X_ = self.separar_classes(X, y)
        self.predic_probabilidade(X_)
        self.medias = {}
        self.desvios = {}
        for class_type in self.classes:
            self.medias[class_type] = np.mean(X_[class_type], axis=0)
            self.desvios[class_type] = self.desvioPadrao(X_[class_type])
        print(self.medias)
    
    def desvioPadrao(self, X):
        X_ = X - X.mean(axis = 0)
        return np.sqrt((sum(X_ * X_)/(len(X))))
    def variancia(self, X): 
        return self.desvioPadrao(X)**2
    def probrabilidade(self, X):
        X_=X-X.mean(axis=0)
        w,m = X.shape
        cov = self.matrizDeCovariancia(X)
        exp = np.exp((-1/2) * X_.T @ X_ @ np.linalg.pinv(cov) ) 
        return (1/(np.linalg.det(cov) **(1/2) * (2*np.pi)**(m/2))) * exp
    def prop2(self, x, media, desvio):
        exp = math.exp(- ((x - media)** 2)/(2* desvio**2))
        return (1/(desvio* np.sqrt(2*np.pi))) * exp
   
    def  predic_probabilidade(self, X):
        self.class_prob = {c:math.log(self.class_freq[c],math.e) for c in self.classes}
        for c in self.classes:
            for i in range(len(self.medias)):
                self.class_prob[c] +=math.log(self.prop2(X[i], self.medias[c][i], self.desvios[c][i]), math.e)
        self.class_prob = {c: math.e**self.class_prob[c] for c in self.class_prob}
        return self.class_prob
    
    
    def matrizDeCovariancia(self, X):
        X_ = X - X.mean(axis=0)
        return X_.T @ X_ / (len(X) -1)
    
    def predict(self, X):
        pred = []
        for x in X:
            pred_class = None
            max_prob = 0
            for c, prob in self.predic_probabilidade(x).items():
                if(prob > max_prob):
                    max_prob = prob
                    pred_class = c
            pred.append(pred_class)
        
        return np.array(pred)


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

nb = NaiveBayesGaussiano()

nb.fit(X_train, y_train)
g = QuadraticDiscriminantAnalysis()
g2 = gaussClf()
g2.fit(X_train, y_train)
g.fit(X_train, y_train)

print(g2.predict(X_test))
print(nb.predict(X_test))
print(g.predict(X_test))

