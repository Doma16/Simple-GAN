import numpy as np
from time import perf_counter

import main as model

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.is_available())

PATH = './models'

vect_dim = 48
gen = model.Generator(vect_dim)
gen.load_state_dict(torch.load(PATH+'/genV1'))
gen.eval()

imgs = []

for _ in range(20):
    noise = torch.randn((1,vect_dim))
    gen_img = gen(noise)


    img_t = gen_img.detach().numpy().squeeze()
    imgs.append(img_t)
    '''
    plt.imshow(img_t)
    plt.show()
    '''

rows = 4
cols = 5
fig, axs = plt.subplots(rows, cols, figsize=(10,10))
axs = axs.flatten()

for i in range(len(imgs)):
    axs[i].imshow(imgs[i], cmap='gray')
    axs[i].axis('off')

plt.style.use('dark_background')
plt.tight_layout()
plt.show()

breakpoint()