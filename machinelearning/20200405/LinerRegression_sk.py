import numpy as np
#sklearn 通过Ctrl + B 查看，再右键show in  explorer  查看文件目录
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error #均方误差
#  训练集
x = [1.1, 1.9, 3.2, 4.1, 5]
y = [2.2, 2.8, 3.9, 5.3, 6]
x = np.reshape(x,(5,1))
y = np.reshape(y,(5,1))

plt.scatter(x,y) #点图
plt.show()

#调用模型
lr = LinearRegression()
#训练模型
lr.fit(x,y)
#计算R2（验证数据）
R2 = lr.score(x,y)
print("R2:" + str(R2))

# 还可以使用均方误差和均方根误差
n = np.shape(x)[0]
print(n)
y_pred = lr.predict(x)
#MSE =(1.0/n) * np.square(y-y_pred).sum()   #MSE:0.029452845204934464
MSE = mean_squared_error(y,y_pred)
RMSE =MSE ** 0.5
print("RMSE:" + str(RMSE))
#画图

plt.scatter(x,y)
plt.plot(x,y_pred)
plt.show()

print(lr.coef_[0])    #d_w

print(lr.intercept_)  #d_b