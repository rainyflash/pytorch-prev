import numpy as np
import torch
import matplotlib.pyplot as plt
 
# prepare dataset
xy = np.loadtxt('./data/diabetes.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])   # 取出前-1列 Feature
y_data = torch.from_numpy(xy[:, [-1]])  # 取最后一列 label
 
# design model using class
 
 
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 32)    # 输入数据x的特征是8维，x有8个特征 #个人修改注释：从 8维升到32维，包含两两组合所有可能的特征
        self.linear2 = torch.nn.Linear(32, 4)    
        self.linear3 = torch.nn.Linear(4, 1)
        self.activation1 = torch.nn.ReLU()       # 将其看作是网络的一层，而不是简单的函数使用
        self.activation2 = None                  #个人修改注释：第一层已经将非线性充分体现，第二层不需要激活函数
        self.activation3 = torch.nn.Sigmoid()    #个人修改注释：输出层使用sigmoid函数
    
    def forward(self, x):
        x = self.activation1(self.linear1(x))
        x = (self.linear2(x))
        x = self.activation3(self.linear3(x))       # y hat
        return x
 
 
model = Model()
 
# construct loss and optimizer
# criterion = torch.nn.BCELoss(size_average = True)
criterion = torch.nn.BCELoss(reduction='mean')  
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #个人修改注释：实测使用Adam优化器效果更好

# 添加学习率调度器（新增代码）
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',      # 监控loss的下降
    factor=0.5,      # 学习率衰减系数
    patience=500,    # 连续500次loss不下降则调整
    verbose=True     # 显示调整信息
)

epoch_list = []
loss_list = []
# training cycle forward, backward, update
for epoch in range(23000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    epoch_list.append(epoch)
    loss_list.append(loss.item())
 
    optimizer.zero_grad()
    loss.backward()
 
    optimizer.step()
    
    # 添加学习率调整（新增代码）
    scheduler.step(loss)  # 根据当前loss调整学习率
 
 
plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()