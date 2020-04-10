# -*-coding:utf-8 -*-
# @Time:2020/3/29 16:52
# @Author:CHM
# @File:decison_tree.py

import numpy as np
from sklearn import tree
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = pd.read_csv("data_clf.csv")
x = data[["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15"]]
y = data[["A16"]]

# 特征工程（简单）
vec = DictVectorizer(sparse=False)
x_dict = x.to_dict(orient="record")
x_vec = vec.fit_transform(x_dict)

# 预处理
sc = StandardScaler()
x_pre = sc.fit_transform(x_vec)

# 分割数据
x_train ,x_test,y_train,y_test = train_test_split(x_pre,y,test_size=0.3,random_state=1)

# 定义SVM
svc = SVC()
svc.fit(x_train,y_train)
score = svc.score(x_test,y_test)
print("SVM:",score)

# 定义决策树
dt = tree.DecisionTreeClassifier(criterion="gini",max_depth=500,random_state=1)
dt.fit(x_train,y_train)
score = dt.score(x_test,y_test)
print("决策树：",score)

