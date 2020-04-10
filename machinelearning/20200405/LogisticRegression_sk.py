from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
n = x.shape[1] # 列，有多少个特征值，即特征维度

#预处理
std = x.std(axis=0)  #标准差
mean = x.mean(axis=0)  #平均值
x = (x-mean)/std # 标准化


# 加1是因为偏置项，如果没有就相当于总是过原点，这样效果会很差
def add_ones(x):  #多增加一个特征列     
    ones = np.ones((x.shape[0],1))  #x.shape[0] 行，有多少数据
    x = np.hstack((ones,x))         # 堆放在一起
    return x

x = add_ones(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=12345)
# solver='liblinear'
# FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
#   FutureWarning)
#实例化
lr = LogisticRegression(solver='liblinear',max_iter=1500)
#训练度
lr.fit(x_train,y_train)
#精确率，召回率，准确率
score_train = lr.score(x_train,y_train)
score_test = lr.score(x_test,y_test)
print("train_score：" + str(score_train))
print("test_score：" + str(score_test))
