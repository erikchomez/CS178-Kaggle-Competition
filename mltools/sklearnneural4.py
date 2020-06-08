import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import sys
sys.path.append('C:\\Users\\radad\\OneDrive\\Desktop\\cs178\\CS178-Kaggle-Competition')
import mltools as ml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from collections import Counter

X = np.genfromtxt('C:\\Users\\radad\\OneDrive\\Desktop\\cs178\\CS178-Kaggle-Competition\\X_train.txt', delimiter=None)
Y = np.genfromtxt('C:\\Users\\radad\\OneDrive\\Desktop\\cs178\\CS178-Kaggle-Competition\\Y_train.txt', delimiter=None)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state=10)

XtrS, params = ml.rescale(X_train)
Xvas, _ = ml.rescale(X_test, params)

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(XtrS, y_train)

mlp = MLPClassifier(solver='sgd', max_iter=2000)
mlp.hidden_layer_sizes = (100, 100, 100)
mlp.activation = 'logistic'
mlp.learning_rate_init = 0.1
mlp.learning_rate = 'adaptive'
mlp.verbose = True

mlp.fit(X_res, y_res)

print(mlp.score(Xvas, y_test))

Xte = np.genfromtxt('C:\\Users\\radad\\OneDrive\\Desktop\\cs178\\CS178-Kaggle-Competition\\X_test.txt', delimiter=None)
Yte = np.vstack((np.arange(Xte.shape[0]), mlp.predict_proba(Xte)[:,1])).T
np.savetxt('C:\\Users\\radad\\OneDrive\\Desktop\\cs178\\CS178-Kaggle-Competition\\Y_submit.txt',Yte,'%d, %.2f',header='ID,Prob1',comments='',delimiter=',')