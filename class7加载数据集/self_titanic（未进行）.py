# 数据加载
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# 准备数据集
class DiabetesDataset(Dataset):                         # 抽象类DataSet
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]                          # shape(多少行，多少列)
        self.x_data = torch.from_numpy(xy[:, :-1])      
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
 
    def __len__(self):
        return self.len

if __name__ == '__main__':
    # dataset对象
    dataset = DiabetesDataset('./data/titanic/train.csv')

# 使用DataLoader加载数据（移动到main函数内部）
if __name__ == '__main__':
    train_loader = DataLoader(dataset=dataset,  # 确保在main函数内初始化
                            batch_size=32,
                            shuffle=True,
                            num_workers=4)  # 多进程需要放在main中

# design model using class
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
 
if __name__ == '__main__':
    model = Model()
    
    # construct loss and optimizer
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    epoch_list = []
    loss_list = []
# training cycle forward, backward, update
if __name__ == '__main__':
    for epoch in range(100):
        total_loss = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data                   # 取出一个batch 
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            # 更新
            optimizer.step()
            
        # 修正平均损失计算（使用实际batch数量）
        avg_loss = total_loss / len(train_loader)  # 替代 dataset.len/32
        print(f"Epoch {epoch}, Average Loss: {avg_loss}")
        epoch_list.append(epoch)
        loss_list.append(avg_loss)

    plt.plot(epoch_list, loss_list)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()