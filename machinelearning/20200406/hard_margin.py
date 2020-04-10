import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt
from sklearn.svm import  SVC #SVC是线性  SVR 是逻辑回归

def load_data():
    data = pd.read_csv("data.txt")
    class_dict = {"class0": 0, "class1": 1, "class2": 2}
    data["class"] = data["class"].map(class_dict)  #class 映射，原来是class会变成0
    X = data[["feature2_length", "feature2_width"]]
    y = data["class"]
    Ture_or_False = (y == 0) | (y == 1)   # 2分类问题，0，1 为Ture  2  为False
    X = X[Ture_or_False]  #数据切分 取0，1 数据
    y = y[Ture_or_False]
    X = np.array(X)
    y = np.array(y)
    return X, y

def plot_svc_boundary(svm, xmin, xmax):
    x0 = np.linspace(xmin, xmax, 250)
    w = svm.coef_[0]
    b = svm.intercept_[0]
    boundary = -w[0] / w[1] * x0 - b / w[1] #W0X0 + w1X1 + b = 0 已知x0,计算X1
    margin = 1 / w[1]
    up = boundary + margin
    down = boundary - margin
    svs = svm.support_vectors_#支持向量
    plt.scatter(svs[:, 0], svs[:, 1], s=1110, facecolors='#FF33AA')
    plt.plot(x0, boundary, "k-", linewidth=3)
    plt.plot(x0, up, "k--", linewidth=2)
    plt.plot(x0, down, "k--", linewidth=2)

def plot_max_margin(X, y):
    svm = SVC(kernel="linear", C=10 ** 10)
    svm.fit(X, y)
    plt.figure(figsize=(8, 4.5))
    plot_svc_boundary(svm, 0, 5.5)  #画3条线
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", label="class1")
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", label="class0")
    plt.xlabel("feature2 length")
    plt.ylabel("feature2 width")
    plt.legend(loc="upper left")
    plt.axis([0, 5.5, 0, 2])
    plt.show()

if __name__=="__main__":
    X, y = load_data()
    plot_max_margin(X, y)
