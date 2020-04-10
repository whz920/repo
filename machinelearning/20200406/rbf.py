# -*-coding:utf-8 -*-
# @Time:2020/3/29 16:52
# @Author:CHM
# @File: rbf.py

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np

def load_data():
    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
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
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)

if __name__ == "__main__":
    X, y = load_data()
    axes = [-2, 3, -1, 2]
    gamma1, gamma2 = 1, 100
    C1, C2 = 1, 100
    hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)
    svms = []
    for gamma, C in hyperparams:
        rbf_kernel_svm = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", gamma=gamma, C=C))
        ])
        rbf_kernel_svm.fit(X, y)
        svms.append(rbf_kernel_svm)
    plt.figure(figsize=(12, 8))
    for i, svm in enumerate(svms):
        plt.subplot(221 + i)
        plot_contourf(svm, axes)
        plot_scatter(X, y, axes)
        gamma, C = hyperparams[i]
        plt.title(r"gamma = {}, C = {}".format(gamma, C))
    plt.show()
