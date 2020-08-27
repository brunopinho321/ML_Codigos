import numpy as np
import re
import Mutivariada_naive_bayes as mnb
import RegressaoLogistica as Rl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import SVC
def plot_confusion_matrix(X_, y_, clf): 
    true  = y_
    classes = np.unique(y_)
    nome = (re.sub(r'^[^.]*.', '', str(clf.__class__)))
    nome = re.sub("'>",'' ,nome)
    pred = clf.predict(X_)
    cm = confusion_Matrix(true, pred)
    fig, ax = plt.subplots()
    titulo = "Matriz de confusão | Classificador: "+nome
    im = ax.imshow(cm, interpolation='nearest', cmap = plt.cm.Blues)
    ax.figure.colorbar(im, ax = ax)
    ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels = classes, yticklabels = classes, title=titulo,
            ylabel = 'True label',
            xlabel =  'Predicted label' )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max()/ 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), str('d')),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
def confusion_Matrix(true, pred):
    b = np.unique(true)
    for i in range(len(true)):
        true[i] = int((true[i]))
        pred[i] = int((pred[i]))
    k = len(np.unique(true))
    result = np.zeros((k,k))
    for i, j in zip(true, pred):
        result[int(i)][int(j)] += 1

    return np.matrix(result)


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

    def predict_prob(self, X):
        p_posteriores = []

        for i, c in enumerate(self.classes):
            anterior = (self.p_anteriores[i])
            posterior = np.prod((self.prob(i, X)))
            posterior *= anterior
            p_posteriores.append(posterior)
        return self.classes[np.argmax(p_posteriores)]     

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.predict_prob(x))
        return np.array(y_pred)
    
    

data = np.loadtxt("C:/Users/Janaina/Desktop/ML_Codigos/NaiveBayes/ex2data1.txt", skiprows=1, delimiter=",")
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
a = mnb.Metricas()
b = mnb.NaiveBayesGaussiano()
b.fit(X_train, y_train)
c = Rl.RegressaoLogistica()
c.fit(X_train, y_train)
a.acuracia(y_test, c.predict(X_test))

plot_confusion_matrix(X_train, y_train, b)
#plot_confusion_matrix(X_train, y_train, c)
#plot_confusion_matrix(X_train, y_train, g)

#plt.show()
