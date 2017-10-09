# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:21:02 2017

@author: GAURAV GAUTAM
"""

import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
start_time = time.time()
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv("C:\\Users\\GAURAV GAUTAM\\Desktop\\datasets.csv", names=names,header=None)
description = data.describe()
print(description)
print((data[[0,1,2,3,4,5,6,7]] == 0).sum())
data['class'].replace([0,1],[-1,1],inplace=True)
# mark zero values as missing or NaN
data[[0,1,2,3,4,5,6,7,8]] = data[[0,1,2,3,4,5,6,7,8]].replace(0, np.nan)
# count the number of NaN values in each column
print(data.isnull().sum())
# fill missing values with mean column values
data=data.fillna(data.mean(), inplace=True)
# count the number of NaN values in each column
print(data.isnull().sum())

values = data.values

#values=data.values
X = values[:, 0:8]
y = values[:,8]
'''plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
plt.xlabel('X1')
plt.ylabel('X2')'''

scaler = MinMaxScaler(feature_range=(0, 8))
rescaledX = scaler.fit_transform(X)
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
print(normalizedX[0:5,:])
# summarize transformed data
np.set_printoptions(precision=3)
print(rescaledX[0:5,:])
lda=LinearDiscriminantAnalysis()
X=lda.fit(X,y).transform(X)
print(X)
#figure = plt.figure(figsize=(27, 9))'''
validation_size = 0.20
seed = 8
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
lda=LinearDiscriminantAnalysis(n_components=2)

X_train = lda.fit_transform(X_train, Y_train)
X_validation = lda.transform(X_validation)
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    { 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']


for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=10,
                       scoring='%s_macro' % score)
    clf.fit(X_train, Y_train)
    
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    print("The best classifier is: ", clf.best_estimator_)
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = Y_validation, clf.predict(X_validation)
    print(classification_report(y_true, y_pred))
    print()
print(accuracy_score(Y_validation, y_pred))
elapsed_time_secs = time.time() - start_time

print ("Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs)))
'''
def plot_svc(svc, X, y, h=0.02, pad=0.25):
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)
    plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
    # Support vectors indicated in plot by vertical lines
    sv = svc.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c='k', marker='x', s=100, linewidths='1')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    print('Number of support vectors: ', svc.support_.size)

plot_svc(clf.best_estimator_, X_validation, Y_validation)
print(confusion_matrix(Y_validation, clf.best_estimator_.predict(X_validation)))
print(clf.best_estimator_.score(X_validation, Y_validation))
'''
