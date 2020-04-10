import numpy as np
from sklearn import tree
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

data = pd.read_csv("data_reg.csv")
# AT（温度）, V（压力）, AP（湿度）, RH（压强）, PE（输出电力)
x = data[["AT","V","AP","RH"]]
y = data[["PE"]]

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
svr = SVR()
svr.fit(x_train,y_train)
score = svr.score(x_test,y_test)
print("SVM:",score)

# 定义决策树
dt = tree.DecisionTreeRegressor(criterion="mse",max_depth=500,random_state=1)

dt.fit(x_train,y_train)
score = dt.score(x_test,y_test)
print("决策树：",score)

