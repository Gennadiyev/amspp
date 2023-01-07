import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils

class SNN(nn.Module):
    def __init__(self,beta=0.9):
        super(SNN, self).__init__()
        self.beta=beta
        self.spike_grad=surrogate.fast_sigmoid()
        self.net=nn.Sequential(
                nn.Conv2d(1, 6, 5),
                nn.MaxPool2d(2),
                snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True),
                nn.Conv2d(6, 16, 5),
                nn.MaxPool2d(2),
                snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True),
                nn.Flatten(),
                nn.Linear(16*4*4, 10),
                snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True, output=True)
        )
                
    def forward(self, x, num_steps=100):
        spk_rec=[]
        utils.reset(self.net)
        for step in range(num_steps):
                spk_out,mem_out=self.net(x)
                spk_rec.append(spk_out)
        return torch.stack(spk_rec)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(        
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),                 
            nn.MaxPool2d(2), 
        )
        self.conv2 = nn.Sequential(        
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(),                 
            nn.MaxPool2d(2), 
        )

        self.fc1 = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(84, 10),
            nn.LeakyReLU(),

        )
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x2 = x2.view(x.size(0), -1)
        x3 = self.fc1(x2)
        x4 = self.fc2(x3)
        x5 = self.fc3(x4)
        return x5


if __name__ == "__main__":
    net = CNN()
    snn_net = SNN()
    a = torch.randn(5, 1, 28, 28)
    print(snn_net(a))
    print(net(a))
    