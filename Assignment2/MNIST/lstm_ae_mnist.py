import torch.optim as optim
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081))])
trainset = torchvision.datasets.MNIST(root="../data/", train=True, download=True, transform=transform)
trainloader = torch.utils.data.Dataloader(trainset, batch_size=100, shuffle=True)
testset = torchvision.datasets.MNIST(root="../data/", train=False, download=True, transform=transform)
testloader = torch.utils.data.Dataloader(testset, batch_size=len(testset), shuffle=False)


batch = 100
epochs = 20
optimizer = torch.optim.Adam
hidden_state_sz = 40
num_layers = 1
lr = 0.01
input_sz = 1
dropout = 0
seq_sz = 50
output_sz = 1
grad_clip = None