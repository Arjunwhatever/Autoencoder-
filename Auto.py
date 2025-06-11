import torch
import torchvision
import torchvision.datasets as dataset
import torch.nn as nn 
from torch.utils.data import DataLoader
import torch.functional as F
import  torchvision.transforms as transforms
from torch.optim import Adam
import matplotlib.pyplot as plt
#from torch import MSELoss
#Data import 

device = torch.device('cuda')
train_data = dataset.MNIST(root = './data', train= True,download= True,transform= transforms.ToTensor())
test_data = dataset.MNIST(root= './data',train= False, download= True, transform= transforms.ToTensor())

train = DataLoader(batch_size= 128,shuffle = True,dataset= train_data)
test = DataLoader(batch_size = 128, shuffle= False, dataset= test_data)
#MNIST = 28*28 = 784
class AutoDec(nn.Module):
    def __init__(self):
        super(AutoDec,self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding= 1 ),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=16,out_channels= 32,kernel_size=3, padding = 1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=32,out_channels= 64,kernel_size=3, padding = 1)
                                     )
         
        self.decoder = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding= 1 ),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=32,out_channels= 16,kernel_size=3, padding = 1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=16 ,out_channels= 1,kernel_size=3, padding = 1),
                                     nn.Sigmoid()
                                     )
        


    def forward(self,x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
    

model = AutoDec().to(device)

criterion = nn.MSELoss()
optim = Adam(model.parameters(),lr = .001,weight_decay = .005)
outputs = []
for epoch in range(10):
     
     for image,_ in train:
          image = image.to(device)
          output = model(image)
          loss = criterion(output, image)
          optim.zero_grad()
          loss.backward()
          optim.step()
     outputs.append((epoch, image, output))
     print(epoch, loss)

for k in range(0, 10, 4):
    plt.figure(figsize=(9, 2))
    plt.gray()
    imgs = outputs[k][1].detach().cpu().numpy()
    recon = outputs[k][2].detach().cpu().numpy()
    for i, item in enumerate(imgs):
        if i >= 9: break
        plt.subplot(2, 9, i+1)
        plt.imshow(item[0])

    for i, item in enumerate(recon):
        if i >= 9: break
        plt.subplot(2, 9, 9+i+1)
        plt.imshow(item[0])
    plt.show()
#Testing 

model.eval()
total_loss = 0
with torch.no_grad():
    for images, _ in test:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            total_loss += loss.item()
    avg_loss = total_loss / len(test)
    print(f"\nTest Reconstruction Loss: {avg_loss:.4f}")