import numpy as np
from time import perf_counter

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets

from torchsummary import summary

from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=2, k=3, bias=True):
        super(BNReLUConv,self).__init__()
        #self.append(nn.BatchNorm2d(in_channels))
        self.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k, stride=stride, bias=bias))
        self.append(nn.LeakyReLU(0.1))

class BNReLUDeConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, k=3, stride=2, bias=True):
        super(BNReLUDeConv,self).__init__()
        #self.append(nn.BatchNorm2d(out_channels))
        self.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k, stride=stride, bias=bias))
        self.append(nn.BatchNorm2d(out_channels))
        self.append(nn.ReLU())  

class Discriminator(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.disc1 = nn.Sequential(
            nn.Dropout2d(),
            BNReLUConv(in_channels,32,stride=1),
            BNReLUConv(32,64,stride=1),
            BNReLUConv(64,64,stride=2),
            BNReLUConv(64,64,stride=1),
            BNReLUConv(64,32,stride=2)
        )
        self.disc2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*4*4,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        y = self.disc1(x)
        #y = y.view(y.shape[0],-1)
        y = self.disc2(y)
        return y

class Generator(nn.Module):
    def __init__(self, vect_dim) -> None:
        super().__init__()
        self.gen1 = nn.Sequential(
            nn.Linear(vect_dim,2*2*vect_dim),
            nn.Unflatten(1,(vect_dim,2,2)),
            BNReLUDeConv(vect_dim,64),
            BNReLUDeConv(64,64),
            BNReLUDeConv(64,64),
            nn.ConvTranspose2d(64,32, kernel_size=6),
            nn.Tanh()
            #BNReLUConv(8,1,k=5)
            #BNReLUConv(12,8,stride=1),
            #BNReLUDeConv(8,1,k=6)
            #nn.Flatten(),
            #nn.Unflatten(1,(1,28,28))
            #nn.Linear(8*14*14,28*28),
        )

    def forward(self,x):
        y = self.gen1(x)
        y = torch.mean(y,dim=(1))
        y = y.view(y.shape[0],1,28,28)
        y = torch.relu(y)

        #y = (y - torch.min(y))#/(torch.max(y) - torch.min(y))
        #y = torch.tanh(y)
        #y = torch.sigmoid(y)
        #y = nn.functional.normalize(y,dim=(2,3))
        #y = torch.mean(x, dim = (0,1))
        return y

if __name__=='__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lrG = 1e-3
    lrD = 4e-4
    vect_dim = 48
    batch_size = 32
    epochs = 50

    in_ch = 1

    fixed_noise = torch.randn((batch_size,vect_dim)).to(device)

    disc = Discriminator(in_ch).to(device)
    gen = Generator(vect_dim).to(device)

    summary(disc, input_size=(1,28,28))
    summary(gen, input_size=(vect_dim,))

    transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    dataset = datasets.MNIST(root='../dataset/', transform=transforms, download=True)
    #dataset.data = dataset.data * 1. / 255.

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt_gen = optim.Adam(gen.parameters(), lr=lrG)
    opt_disc = optim.Adam(disc.parameters(), lr=lrD)

    loss = nn.BCELoss()

    writer_fake = SummaryWriter(f'runs/ConvGAN/fake')
    writer_real = SummaryWriter(f'runs/ConvGAN/real')
    step = 0

    for epoch in range(epochs):
        start = perf_counter()
        for batchidx, (real, _) in enumerate(loader):

            real = real.to(device)
            batch_size = real.shape[0]


            ### train Disc : max log(D(real)) + log(1 - D(G(z)))

            disc_real = disc(real).view(-1)
            lossD_real = loss(disc_real, torch.ones_like(disc_real))

            noise = torch.randn((batch_size,vect_dim)).to(device)
            fake = gen(noise)
            disc_fake = disc(fake).view(-1)
            lossD_fake = loss(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2

            disc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            ### train Gen : min log(1 - D(G(z))) -> max log (D(G(z)))

            output = disc(fake).view(-1)
            lossG = loss(output, torch.ones_like(output))

            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            if batchidx == 0:
                print(
                    f'Epoch [{epoch}/{epochs}] \ ' 
                    f'Loss D: {lossD:.4f}, Loss G: {lossG:.4f}'
                )

                with torch.no_grad():
                    fake = gen(fixed_noise)
                    data = real

                    #print(fake[0].squeeze())
                    #print('-------------')
                    #print(data[0].squeeze())
                    

                    img_grid_real = torchvision.utils.make_grid(real,normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake,normalize=True)

                    writer_real.add_image(
                        "Mnist Real Images", img_grid_real, global_step=step
                    )

                    writer_fake.add_image(
                        "Mnist Fake Images", img_grid_fake, global_step=step
                    )

                    step+=1
        end = perf_counter()
        Etime = end - start
        print(f'Epoch time: {Etime}')


    PATH = './models'
    torch.save(disc.state_dict(), PATH+'/disc')
    torch.save(gen.state_dict(), PATH+'/gen')