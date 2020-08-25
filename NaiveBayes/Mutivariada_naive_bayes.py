import numpy as np
import math
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from scipy.stats import multivariate_normal
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

    def matrizDeCovariancia(self, X):
        X_ = X - X.mean(axis=0) 
        m = X_.T @ X_ / (len(X) -1)
        m_x = m
        a,b = m_x.shape
        for i in range(a):
            for j in range(b):
                if i != j:
                    m_x[i][j] = 0
        return m_x
    def probabilidade(self, x, media, covariancia):
        exp = np.exp(((x-media).T * np.linalg.inv(covariancia) @ (x-media))) * (-1/2)
        p = ((1/((np.linalg.det(covariancia) **(1/2)) * ((2*np.pi)**(len(x)/2))))) * exp
        return np.sum((p))
    
    def fit(self, X, y):
        self.n = len(X)
        self.X_= self.separar_classes(X,y)
        self.medias = {}
        self.covariancias = {}
        self.p_anteriores = {}
        for i in self.classes:
            self.medias[i] = np.mean(self.X_[i], axis= 0)
            self.covariancias[i] = self.matrizDeCovariancia(self.X_[i])
            self.p_anteriores[i] = self.X_[i].shape[0]/self.n

    def  predic_probabilidade(self, X):
        c = {}
        a = []
        p = {}
        for i in self.classes:
            for x in X:
                a.append(np.array(self.probabilidade(x, self.medias[i], self.covariancias[i])))
            c[i] = np.array(a)
            a = []
        #print(c)
        for i in self.classes:
            p[i] = len(self.X_[i])/self.n
       
        
        for i in self.classes:
            c[i] = ((c[i] * np.array([[p[i]]])))
        
        a = c[0][0]
        a = np.c_[a, c[1][0]]
       # print(a)
        b = []
        for i in a:
           b.append(np.argmax(i)*1.0)
        print(np.array(b))

    
    def predict_prob(self, x):
        m = []
        m1 = []
        for i in range(len(self.medias)):
            posterior = (((self.probabilidade(x, self.medias[i], self.covariancias[i]))))
            anterior =(self.p_anteriores[i]) 
            posterior *= anterior
            m.append(posterior)
        return(self.classes[np.argmax(m)])
       
    def predict2(self, X):
        y_pred = []
        for x in X:
           y_pred.append(self.predict_prob(x))
        print(np.array(y_pred))

    def predict(self, X):
        m = []
        m1 = []
        indice = -np.inf
        for x in X:
            for i in range(len(self.medias)):
                a = (self.probabilidade(x, self.medias[i], self.covariancias[i]))
                m.append(a)
            m1.append(m)
            m =[] 
        for i in m1:
            m.append(np.argmax(i))
        print(np.array(m ) * 1.0)
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
nb.fit(X_train,y_train)
nb.predict2(X_test)
nb.predict(X_test)
nb.predic_probabilidade(X_test)
q = QuadraticDiscriminantAnalysis()
q.fit(X_train, y_train)
print(q.predict(X_test))
g = GaussianNB()
g.fit(X_train, y_train)
print(g.predict(X_test))

