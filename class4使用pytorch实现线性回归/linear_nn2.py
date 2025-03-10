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

model = LinearModel()                       #实例化模型

criterion = torch.nn.MSELoss(size_average=False)
#model.parameters()会扫描module中的所有成员，如果成员中有相应权重，那么都会将结果加到要训练的参数集合上
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)#lr为学习率

# 在训练前初始化记录列表
epoch_list = []
loss_list = []

for epoch in range(101):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    
    # 记录每个epoch的loss
    epoch_list.append(epoch)
    loss_list.append(loss.item())
    
    if epoch % 10 == 0:  # 每10次打印一次
        print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 训练结束后绘制曲线
plt.plot(epoch_list, loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)
