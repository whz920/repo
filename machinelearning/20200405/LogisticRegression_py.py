import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()


data = pd.DataFrame(data = dataset.data,columns =dataset.feature_names)
data['cancer'] =[dataset.target_names[t] for t in dataset.target]
print(dataset.target_names)
print(data['cancer'].value_counts())


def sigmoid(z):
    s = 1/(1 + np.exp(-z))
    s = s.reshape(s.shape[0],1)
    return s
# 画Sigmoid图，注释多行 Ctrl + / ,不要在五笔的模式下，有冲突
# def draw_sigmoid():
#     x = np.arange(-6,6,0.01)
#     y=sigmoid(x)
#     plt.plot(x,y,color ='red')
#     plt.show()
# draw_sigmoid()
#定义模型 (θ^T * X)
def model(theta,x):
    #z = np.sum(theta.T * x,axis =1) # theta.T 是转置
    z = np.dot(x,theta)
    return sigmoid(z)

#定义交叉熵
def cross_entropy(y,y_pred):
    m = y.shape[0] # m =np.shape(y)[0]
    return sum(-y*np.log(y_pred)-(1-y)*np.log(1-y_pred))/m


#损失函数
def cost(theta,x,y):
    y_pred = model(theta,x)
    return cross_entropy(y,y_pred)

#优化器
def optimize(theta,x,y):
    m = x.shape[0]
    alpha = 1e-1
    y_pred = model(theta,x)
    d_theta =(1.0/m) *((y_pred-y)*x)  #求导，然后求和
    d_theta =np.sum(d_theta,axis =0) #列模式,形成一维
    d_theta = d_theta.reshape((31,1)) # 31个特征值，即特征维度,转为列
    theta = theta - alpha * d_theta
    return theta

def predict_proba(theta,x):
    y_pred = model(theta,x)
    return y_pred

def predict(x,theta):
    y_pred=predict_proba(theta,x)
    y_hard=(y_pred > 0.5)  * 1
    return y_hard

def accuracy(theta,x,y):
    y_hard=predict(x,theta)
    count_right=sum(y_hard == y)
    return count_right*1.0/len(y)


#迭代器
def iterate(theta,x,y,x_test,y_test,n):
    costs = []
    acc_train = []
    acc_test = []
    for i in range(n):
        theta = optimize(theta,x,y)
        costs.append(cost(theta,x,y))
        acc_train.append(accuracy(theta,x,y))
        acc_test.append(accuracy(theta,x_test,y_test))
    return theta,costs,acc_train,acc_test

x = dataset.data
y = dataset.target
n_features = x.shape[1]  # 列，有多少个特征值，即特征维度

# 预处理
std = x.std(axis=0)  # 标准差
mean = x.mean(axis=0)  # 平均值
x = (x - mean) / std  # 标准化

def add_ones(x):  # 多增加一个特征列
    ones = np.ones((x.shape[0], 1))  # x.shape[0] 行，有多少数据
    x = np.hstack((ones, x))  # 堆放在一起
    return x

x = add_ones(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12345)
y_train =y_train.reshape((y_train.shape[0],1))
y_test = y_test.reshape((y_test.shape[0],1))


theta = np.ones((n_features+1,1))
theta, costs,acc_train,acc_test = iterate(theta,x_train,y_train,x_test,y_test,1500)
plt.plot(acc_train,color='green')
plt.plot(acc_test,color='red')
plt.show()
#最后一次精确度
print(costs[-1],acc_train[-1],acc_test[-1])

