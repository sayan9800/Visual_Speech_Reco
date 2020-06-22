import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

filename = 'Thank_You.txt'

#Test

#train = np.loadtxt(filename, delimiter = ',', dtype = 'float32')
full = np.arange(1, 10001, 1, dtype = 'float32')
full = full.reshape(-1, 100)
full_dataset = torch.from_numpy(full)


train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_dataset = DataLoader(train_dataset, batch_size = 1, shuffle = True)
test_dataset = DataLoader(test_dataset, batch_size = 1, shuffle = True)
#Testing the x-axis data

#Hyperparameters
num_epochs = 100
batch_size = 128
learning_rate = 1e-3

class Autoencoder(nn.Module):
    def __init__(self, size, hidden1 = 64, hidden2 = 32, hidden3 = 16, hidden4 = 8):
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
model = Autoencoder(size = 100)
loss_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
	for train in train_dataset:
	    #foreward
	    output = model(train)
	    loss = loss_criterion(output, train)

	    #backward
	    optimizer.zero_grad()
	    loss.backward()
	    optimizer.step()
	
	print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
	loss_data.append(loss.item())

for test in test_dataset:
	out = model(test)
	loss = loss_criterion(out, test)
	print(test)
	print(out)
	print(model.encoder(test))
	print(loss)
	print('\n')

# print(train_tensor)
# print(output)
# print('\n')
# encoded_tensor = model.encoder(train_tensor)
# print(encoded_tensor, len(encoded_tensor))
# plt.plot(range(num_epochs), loss_data)
# plt.ylabel('MSE LOSS')
# plt.xlabel('epochs')
# plt.grid(True)
# plt.show()
