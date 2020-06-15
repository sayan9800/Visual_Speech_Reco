import os
import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import transforms

filename = 'Thank_You.txt'

train = np.loadtxt(filename, delimiter = ',', dtype = 'float32')
train = train.reshape(-1, 13, 2)
#print(train[:,0,:])
train_tensor = torch.from_numpy(train)
print(train_tensor)

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder() #.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    #for data in dataloader:
    img = train_tensor
    img = img.view(img.size(0), -1)
    img = Variable(img) #.cuda()
    print(img)
        # ===================forward=====================
    output = model(img)
    loss = criterion(output, img)
        # ===================backward====================
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # ===================log========================
    #print('epoch [{}/{}], loss:{:.4f}'
          #.format(epoch + 1, num_epochs, loss.data[0]))