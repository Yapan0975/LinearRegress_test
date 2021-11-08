# from typing import OrderedDict
import numpy as np
import torch

import matplotlib.pyplot as plt
from torch._C import device
from torch.nn.modules import module

np.random.seed(42)

observations = 1000

x = np.random.uniform(low=-10, high=10, size= (observations, 1))

print(x.shape)

noise = np.random.uniform(-1, 1, (observations,1))
targets = 13*x+2 +noise
print(targets.shape)

plt.plot(x,targets)
plt.ylabel("Targets")
plt.xlabel("Input")
plt.title("Data")
plt.show()

from torch.utils.data import TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(targets).float()

train_data = TensorDataset(x_tensor, y_tensor)

torch.manual_seed(42)

def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.uniform_(m.weight, a = -0.1, b = 0.1)
        torch.nn.init.uniform_(m.bias, a = -0.1, b = 0.1)

model = torch.nn.Sequential(torch.nn.Linear(1, 1)).to(device)

model.apply(init_weight)
print(model.state_dict())
lr = 0.02
loss_fn = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


from torch.utils.data import DataLoader
train_loader = DataLoader(train_data, batch_size = 16, shuffle= True)

def make_train_step(model, optimizer, loss_fn):
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(y, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()
    return train_step

train_step = make_train_step(model, optimizer, loss_fn)
losses = []
epochs = 100

for epoch in range(epochs):
    batch_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        loss = train_step(x_batch, y_batch)
        batch_loss = batch_loss+ loss
    
    epoch_loss = batch_loss / 63
    # 1000/16 =62.5
    losses.append(epoch_loss)

print(model.state_dict())


