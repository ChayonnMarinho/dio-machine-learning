import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets,transforms
from torch import nn,optim

# definindo a conversão de imagem para tensor
transform = transforms.ToTensor() 

# carrega a parte de treino do dataset
trainset = datasets.MNIST('./MNIST_data/',download=True,train=True,transform=transform) 
# cria um buffer para pegar os dados por partes
trainloader = torch.utils.data.DataLoader(trainset,batch_size=24,shuffle=True)

# carrega a parte de validação do dataset
valset = datasets.MNIST('./MNIST_data/',download=True,train=False,transform=transform)
# cria um buffer para pegar os dados por partes
valloader = torch.utils.data.DataLoader(trainset,batch_size=24,shuffle=True)

dataiter = iter(trainloader)
imagens, etiquetas = next(dataiter)
plt.imshow(imagens[1].numpy().squeeze(),cmap='gray_r')
plt.show()

# para verificar as dimensões do tensor de cada imagem
print(imagens[1].shape)
#para verificar as dimensões do tensor de cada etiqueta
print(etiquetas[1].shape)