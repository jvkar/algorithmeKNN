import sklearn.model_selection


from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from KNN import KNN



dataset = pd.read_csv('diabetes.csv')


X = dataset.drop('Outcome', axis=1).values
y = dataset['Outcome'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, stratify=y)


classificateur = KNN(k=5)
classificateur.fit(X_train, y_train)
predictions = classificateur.predict(X_test)

print(predictions)

acc = np.sum(predictions == y_test) / len(y_test)
print("Accurrence:", acc)
