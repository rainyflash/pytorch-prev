# 绘图
import numpy as np              
import matplotlib.pyplot as plt 
 
# 数据集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 模型 前馈
def forward(x):
    return x*w
# 损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2

# 保存权重和损失
w = np.arange(0.0, 4.1, 0.1)
l_sum = 0

for x_val, y_val in zip(x_data, y_data):
    # 计算预测
    y_pred_val = forward(x_val)
    # 计算损失
    loss_val = loss(x_val, y_val)
    # 求和
    l_sum += loss_val
    print('\t', x_val, y_val, y_pred_val, loss_val)
print('MSE=', l_sum/3)
# 添加到列表

# 绘制图形
plt.plot(w,l_sum/3)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()