import matplotlib.pyplot as plt
import numpy as np

# 分治算法
# 数据集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
# initial guess of weight 
# 初始权重
begin = -50
end = 50
step = 10
w = np.arange(begin , end , step)

# 前馈
# define the model linear model y = w*x
def forward(x):
    return x*w

# 定义损失代价函数
#define the cost function MSE 
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs,ys):
        y_pred = forward(x)
        cost += (y_pred - y)**2
    cost = cost / len(xs)
    # 找到最小cost对应的索引
    min_idx = np.argmin(cost)
    return cost, min_idx  # 返回cost数组和最小值索引

# 定义分治算法，找出cost最小的点的w
def fenzhi(best_w,step):
    new_step = step / 10
    start = best_w - 50*new_step
    end = best_w + 50*new_step
    w = np.arange(start, end, new_step)
    return w
 
# 记录
epoch_list = []
cost_list = []
print('predict (before training)', 4, forward(4))

# 训练
for epoch in range(100):
    cost_val, min_idx = cost(x_data, y_data)  # 获取最小索引
    best_w = w[min_idx]  # 通过索引获取最佳w值
    w = fenzhi(best_w,step)  # 需要调整fenzhi函数以基于best_w生成新范围
    
    # 更新权值
    print('epoch:', epoch, 'w=', best_w, 'loss=', cost_val[min_idx])
    epoch_list.append(epoch)
    cost_list.append(cost_val[min_idx])  # 记录最小cost值

# 绘图
print('predict (after training)', 4, forward(4))
plt.plot(epoch_list,cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show() 