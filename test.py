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

PATH = './models'

vect_dim = 48
gen = model.Generator(vect_dim)
gen.load_state_dict(torch.load(PATH+'/genV1'))
gen.eval()


for _ in range(100):
    noise = torch.randn((1,vect_dim))
    gen_img = gen(noise)

    plt.imshow(gen_img.detach().numpy().squeeze())
    plt.show()