import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def load_data():
    data = pd.read_csv("data.txt")
    class_dict = {"class0": -1, "class1": -1, "class2": 1}
    data["class"] = data["class"].map(class_dict)
    X = data[["feature2_length", "feature2_width"]]
    y = data["class"]
    X = np.array(X)
    y = np.array(y)
    return X, y

def plot_svc_boundary(svm_clf, xmin, xmax,w,b):
    x0 = np.linspace(xmin, xmax, 200)
    boundary = -w[0] / w[1] * x0 - b / w[1]
    margin = 1 / w[1]
    up = boundary + margin
    down = boundary - margin
    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FF55AA')
    plt.plot(x0, boundary, "k-", linewidth=3)
    plt.plot(x0, up, "k--", linewidth=2)
    plt.plot(x0, down, "k--", linewidth=2)

def plot_margin_with_outlier(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    svm_1 = SVC(kernel="linear", C=20)
    svm_2 = SVC(kernel="linear", C=1000000)
    svm_1.fit(X, y)
    svm_2.fit(X, y)
    b1 = svm_1.intercept_[0]
    b2 = svm_2.intercept_[0]
    w1 = svm_1.coef_[0]
    w2 = svm_2.coef_[0]
    support_vectors_1 = (y * (X.dot(w1) + b1) < 1).ravel()
    support_vectors_2 = (y * (X.dot(w2) + b2) < 1).ravel()
    svm_1.support_vectors_ = X[support_vectors_1]
    svm_2.support_vectors_ = X[support_vectors_2]
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^", label="class0")
    plt.plot(X[:, 0][y == -1], X[:, 1][y == -1], "bs", label="class1")
    plot_svc_boundary(svm_1, -1, 1,w1,b1)
    plt.xlabel("feature2 length")
    plt.legend(loc="upper left")
    plt.ylabel("feature2 width")
    plt.title("$C = {}$".format(svm_1.C))
    plt.axis([-1, 1, -1, 1])
    plt.subplot(122)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^",label="class0")
    plt.plot(X[:, 0][y == -1], X[:, 1][y == -1], "bs", label="class1")
    plot_svc_boundary(svm_2, -1, 1,w2,b2)
    plt.xlabel("feature2 length")
    plt.legend(loc="upper left")
    plt.title("$C = {}$".format(svm_2.C))
    plt.axis([-1, 1, -1, 1])
    plt.show()

if __name__ == "__main__":
    X, y = load_data()
    plot_margin_with_outlier(X, y)
