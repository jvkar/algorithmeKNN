import numpy as np
from collections import Counter

def distance_ecludienne(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):

        distances = [distance_ecludienne(x, x_train) for x_train in self.X_train]
    

        indicesK = np.argsort(distances)[:self.k]
        k_labels_proches = [self.y_train[i] for i in indicesK]


        le_plus_courant = Counter(k_labels_proches).most_common()
        return le_plus_courant[0][0]