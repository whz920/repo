# -*-coding:utf-8 -*-
# @Time:2020/3/29 16:52
# @Author:CHM
# @File: polynomial.py

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def load_data():
    X, y = make_moons(n_samples=250, noise=0.14, random_state=1)
    return X, y

def plot_scatter(X, y, axes):
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
    plt.axis(axes)

def plot_contourf(clf, axes):
    x0 = np.linspace(axes[0], axes[1], 100)
    x1 = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0, x1)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, alpha=0.4)

def svm(X, y, d, r, c):
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="poly", degree=d, coef0=r, C=c))
    ])
    svm.fit(X, y)
    return svm

if __name__ == "__main__":
    X, y = load_data()
    axes = [-2, 3, -1, 2]
    svm_1 = svm(X, y, 3, 1, 5)
    svm_2 = svm(X, y, 10, 100, 5)
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plot_contourf(svm_1, axes)
    plot_scatter(X, y, axes)
    plt.title("d=3, r=1, C=5")
    plt.subplot(122)
    plot_contourf(svm_2, axes)
    plot_scatter(X, y, axes)
    plt.title("d=10, r=100, C=5")
    plt.show()