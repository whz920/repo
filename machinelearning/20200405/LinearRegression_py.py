import numpy as np
import matplotlib.pyplot as plt

#定义模型(前向传播)
def model(x,w,b):
    return  w*x + b

#定义损失函数
def cost(x,y,w,b):
    m = np.shape(x)[0]
    costs = 0.5/m * np.square(w*x+b-y).sum()
    return costs
#优化器
def optimize(x,y,w,b):
    m = np.shape(x)[0]
    alpha = 1e-1  # 0.1
    y_pred = model(x,w,b)
    d_w = (1.0/m) * ((y_pred - y) * x).sum()
    d_b = (1.0/m) * ((y_pred-y).sum())
    w = w - alpha * d_w
    b = b - alpha * d_b
    return w,b
#迭代器
def iterate(x,y,w,b,n):
    for i in range(n):
        w,b = optimize(x,y,w,b)  # w,b 重新赋值
    y_pred = model(x,w,b)
    cost_ = cost(x,y,w,b)
    print(w,b,cost_)
    plt.scatter(x,y)  #点图
    plt.plot(x,y_pred) #线图
    plt.show()
    return w,b




def train(x,y,n):
    w = 0
    b = 0
    w,b = iterate(x,y,w,b,n)
    return w,b

def validate(w,b,x,y):
    y_pred = model(x,w,b)
    y_mean = y.mean()
    SSR = np.square(y_pred-y_mean).sum()
    SST = np.square(y-y_mean).sum()
    R2 =SSR/SST
    print(R2)

if __name__=='__main__':
#  训练集
    x = [1.1, 1.9, 3.2, 4.1, 5]
    y = [2.2, 2.8, 3.9, 5.3, 6]
    x = np.reshape(x,(5,1))
    y = np.reshape(y,(5,1))

    plt.scatter(x,y) #点图
    plt.show()
#训练集
    w,b = train(x,y,10000)
#验证集
    x1 = [1.2, 1.0, 3.5, 4.1, 5.5]
    y1 = [2.5, 1.8, 4.2, 5.3, 6]
    x1 = np.reshape(x,(5,1))
    y1 = np.reshape(y,(5,1))
    validate(w,b,x1,y1)
#测试集