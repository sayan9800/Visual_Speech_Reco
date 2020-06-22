import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt

filename = 'Thank_You.txt'

#Test

train = np.loadtxt(filename, delimiter = ',', dtype = 'float32')
#Testing the x-axis data
train_tensor = torch.from_numpy(train[:,0])

#Hyperparameters
num_epochs = 150
batch_size = 128
learning_rate = 1e-3

class Autoencoder(nn.Module):
    def __init__(self, size, hidden1 = 128, hidden2 = 64, hidden3 = 32, hidden4 = 16):
        super().__init__()
        self.encoder = nn.Sequential(
                        nn.Linear(size, hidden1),
                        nn.ReLU(True),
                        nn.Linear(hidden1, hidden2),
                        nn.ReLU(True),
                        nn.Linear(hidden2, hidden3),
                        nn.ReLU(True),
                        nn.Linear(hidden3, hidden4))

        self.decoder = nn.Sequential(
                        nn.Linear(hidden4, hidden3),
                        nn.ReLU(True),
                        nn.Linear(hidden3, hidden2),
                        nn.ReLU(True),
                        nn.Linear(hidden2, hidden1),
                        nn.ReLU(True),
                        nn.Linear(hidden1, size))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

loss_data = []
model = Autoencoder(size = len(train_tensor))
loss_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    #foreward
    output = model(train_tensor)
    loss = loss_criterion(output, train_tensor)

    #backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #log
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    loss_data.append(loss.item())

print(train_tensor)
print(output)
print('\n')
encoded_tensor = model.encoder(train_tensor)
print(encoded_tensor, len(encoded_tensor))
plt.plot(range(num_epochs), loss_data)
plt.ylabel('MSE LOSS')
plt.xlabel('epochs')
plt.grid(True)
plt.show()
