import torch
import torch.nn as nn
import numpy as np
from torch import optim
import cupy as cp
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
# def get_grad(module, grad_in, grad_out):
#     print(module)  # 输出当前层的信息
#     print(grad_in)  # 输出当前层的梯度信息
#     print(grad_out)  # 输出当前层的输出信息
#     print('Grad norm: ', grad_in[0].norm())

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(15, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

    def fit(self, x, y, epochs=25, batch_size=32, lr=0.01):
        # 将numpy数组转换为PyTorch张量
        x = torch.from_numpy(x).float().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        # 定义损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = optim.SGD(self.parameters(), lr=lr)

        # 训练网络
        for epoch in range(epochs):
            # 将数据分成小批次进行训练
            for i in range(0, len(x), batch_size):
                # 获取小批次数据
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                y_batch = y_batch.reshape(-1, 1)
                # 前向传播，计算损失，反向传播
                optimizer.zero_grad()
                output = self(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

            # 计算训练集上的损失并记录
            output = self(x)
            y = y.reshape(-1, 1)
            train_loss = criterion(output, y)
            print(f"Epoch {epoch+1}/{epochs}, Training loss: {train_loss.item():.4f}")

    def predict(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        outputs = self(x)
        outputs = torch.Tensor.cpu(outputs)
        y_pred = outputs.detach().numpy().reshape(-1)

        y_pred[y_pred > 0.3] = 1
        y_pred[y_pred <= 0.3] = 0

        return y_pred


# x = np.random.uniform(0, 600, size=(2000, 20))
#
# label = np.zeros((2000,), dtype=int)
# set_one = np.random.choice(2000, size=int(2000 * 0.1), replace=False)
# label[set_one] = 1
#
# mlp = MLP()


# for name, layer in mlp.named_modules():
#     layer.register_backward_hook(get_grad)
# mlp.fit(x, label)
# print("over")
#
# x_test = np.random.uniform(0, 600, size=(10, 20))


