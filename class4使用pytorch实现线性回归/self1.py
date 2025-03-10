import  torch
import matplotlib.pyplot as plt  # 新增matplotlib导入

x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):                     #构造函数
        super(LinearModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)  #构造对象，并说明输入输出的维数，第三个参数默认为true，表示用到b
    def forward(self, x):
        y_pred = self.linear(x)             #可调用对象，计算y=wx+b
        return  y_pred

# 定义不同优化器配置
optimizers = {
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam,
    "RMSprop": torch.optim.RMSprop,
    "Adagrad": torch.optim.Adagrad
}

# 存储各优化器的训练记录
history = {}
colors = ['r', 'g', 'b', 'm']  # 不同优化器颜色

# 遍历所有优化器
for idx, (opt_name, Opt) in enumerate(optimizers.items()):
    # 每次重新初始化模型
    model = LinearModel()
    criterion = torch.nn.MSELoss()
    
    # 创建优化器（统一使用0.01学习率）
    optimizer = Opt(model.parameters(), lr=0.01)
    
    # 训练记录
    epoch_list = []
    loss_list = []
    
    # 训练循环
    for epoch in range(1000):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        
        # 记录数据
        if epoch % 10 == 0:
            epoch_list.append(epoch)
            loss_list.append(loss.item())
        
        # 优化步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 保存当前优化器的训练结果
    history[opt_name] = (epoch_list, loss_list, colors[idx])

# 绘制对比曲线
plt.figure(figsize=(10, 6))
for name, (epochs, losses, color) in history.items():
    plt.plot(epochs, losses, color, alpha=0.7, label=name)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Optimizer Comparison')
plt.legend()
plt.grid(True)
plt.show()

print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)
