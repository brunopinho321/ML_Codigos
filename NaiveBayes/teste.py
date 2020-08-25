
import numpy as np   

class GDAClassifier:
    
    def fit(self, X, y, epsilon = 1e-10):
        self.y_classes, y_counts = np.unique(y, return_counts=True)
        self.phi_y = 1.0 * y_counts/len(y)
        self.u = np.array([ X[y==k].mean(axis=0) for k in self.y_classes])
        self.E = self.compute_sigma(X, y)
        self.E += np.ones_like(self.E) * epsilon # fix zero overflow
        self.invE = np.linalg.pinv(self.E)
        return self
    
    def compute_sigma(self,X, y):
        X_u = X.copy().astype('float64')
        for i in range(len(self.u)):
            X_u[y==self.y_classes[i]] -= self.u[i]
        return X_u.T.dot(X_u) / len(y)

    def predict(self, X):
        return np.apply_along_axis(self.get_prob, 1, X)
    
    def score(self, X, y):
        return (self.predict(X) == y).mean()
    
    def get_prob(self, x):
        p = np.exp(-0.5 * np.sum((x - self.u).dot(self.invE) * (x - self.u), axis =1)) * self.phi_y
        print(p)
        return np.argmax(p)


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

nb = GDAClassifier()
nb.fit(X_train,y_train)
nb.predict(X_test)