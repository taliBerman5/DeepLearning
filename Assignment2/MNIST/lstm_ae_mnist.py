import torch.optim as optim
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

from Assignment2.LSTM_AE import LSTM_AE as AE
import math

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081))])
trainset = torchvision.datasets.MNIST(root="../data/", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
testset = torchvision.datasets.MNIST(root="../data/", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

batch = 100
epochs = 20
optimizer = torch.optim.Adam
hidden_state_sz = 20
num_layers = 1
lr = 0.01
input_sz = 28
dropout = 0
seq_sz = 28
output_sz = 28
grad_clip = None


class AE_MNIST():
    def __init__(self, hidden_state_sz, lr, grad_clip):
        self.train_loader = trainloader
        self.test_loader = testloader
        self.lr = lr
        self.batch = batch
        self.epochs = epochs
        self.hidden_state_sz = hidden_state_sz
        self.input_sz = input_sz
        self.seq_sz = seq_sz
        self.output_sz = output_sz
        self.grad_clip = grad_clip
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.auto_encoder = AE(self.input_sz, self.hidden_state_sz, self.num_layers, self.dropout, self.seq_sz,
                               self.output_sz)
        self.optimizer = optimizer(self.auto_encoder.parameters(), lr=self.lr)

    def train(self):
        model = self.auto_encoder.to(self.device)
        criterion = torch.nn.MSELoss().to(self.device)
        stepper = torch.optim.lr_scheduler.StepLR(self.optimizer, 100, 0.5)

        train_loss = []

        for epoch in range(self.epochs):
            print(f'started epoch {epoch}')

            curr_loss = 0
            for b, (img, label) in enumerate(self.train_loader):
                train_ind = img.squeeze().to(self.device)
                self.optimizer.zero_grad()

                # forward pass
                outputs = model(train_ind)
                loss = criterion(outputs, train_ind)

                # backward pass
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                self.optimizer.step()
                curr_loss += loss.item()

            stepper.step()
            train_loss.append(curr_loss / len(self.train_loader))

        return train_loss

    def reconstruct(self, data):
        return self.auto_encoder.to(self.device).forward(data.to(self.device))

    def plot(self):
        train_loss = self.train()
        x = np.arange(self.epochs)
        plt.plot(x, train_loss, label="Train loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(
            f'MNIST LSTM loss \n hidden state size = {self.hidden_state_sz}, learning rate = {self.lr}, gradient clipping = {self.grad_clip}')
        plt.show()

        test_iter = iter(self.test_loader)
        test_images, test_labels = test_iter.next()
        reconstruction = self.reconstruct(test_images[:2]).detach().cpu().squeeze().numpy()

        for i in range(3):
            plt.plot(test_images[i], label="original signal")
            plt.plot(reconstruction[i], label="reconstruction")
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Signal Value")
            plt.title(f'Original vs Reconstruction of example {i}')
            plt.show()


ae = AE_MNIST(hidden_state_sz=hidden_state_sz, lr=lr, grad_clip=grad_clip)
ae.plot()
