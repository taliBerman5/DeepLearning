import torch.optim as optim
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

from Assignment2.LSTM_AE import LSTM_AE as AE
from Assignment2.MNIST.MNIST_LSTM_AE import LSTM_AEC as AEC
import math

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081))])
trainset = torchvision.datasets.MNIST(root="./data/", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
testset = torchvision.datasets.MNIST(root="./data/", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

batch = 100
epochs = 10
optimizer = torch.optim.Adam
hidden_state_sz = 20
num_layers = 1
lr = 0.01
input_sz = 28
dropout = 0
seq_sz = 28
output_sz = 28
input_sz_pixel = 1
seq_sz_pixel = 784
output_sz_pixel = 1
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
        self.input_sz_pixel = input_sz_pixel
        self.output_sz_pixel = output_sz_pixel
        self.seq_sz_pixel = seq_sz_pixel
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.AE = AE(self.input_sz, self.hidden_state_sz, self.num_layers, self.dropout, self.seq_sz,
                     self.output_sz)
        self.optimizer = optimizer(self.AE.parameters(), lr=self.lr)
        self.AEC = AEC(self.input_sz, self.hidden_state_sz, self.num_layers, self.dropout, self.seq_sz,
                       self.output_sz)
        self.optimizer_aec = optimizer(self.AEC.parameters(), lr=self.lr)
        self.AEC_pixel = AEC(self.input_sz_pixel, self.hidden_state_sz, self.num_layers, self.dropout,
                             self.seq_sz_pixel,
                             self.output_sz_pixel)
        self.optimizer_aec_pixel = optimizer(self.AEC_pixel.parameters(), lr=self.lr)

    def train(self):
        model = self.AE.to(self.device)
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

    def train_classification(self, isRow):
        model = self.AEC.to(self.device) if isRow else self.AEC_pixel.to(self.device)
        criterion_MSE = torch.nn.MSELoss().to(self.device)
        criterion_CE = torch.nn.CrossEntropyLoss().to(self.device)
        optimizer = self.optimizer_aec if isRow else self.optimizer_aec_pixel
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 100, 0.5)

        test_iter = iter(self.test_loader)
        test_images, test_labels = test_iter.next()
        test_ind = test_images.squeeze().to(self.device)
        test_labels = test_labels.to(self.device)
        test_ind = test_ind if isRow else test_ind.view(test_ind.shape[0], self.seq_sz_pixel, -1)

        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []

        for epoch in range(self.epochs):
            print(f'started epoch {epoch}')

            curr_loss = 0
            curr_acc = 0

            for b, (images, labels) in enumerate(self.train_loader):
                train_ind = images.squeeze().to(self.device)
                labels = labels.to(self.device)
                train_ind = train_ind if isRow else train_ind.view(train_ind.shape[0], self.seq_sz_pixel, -1)
                optimizer.zero_grad()

                # forward pass
                outputs, classification = model(train_ind)
                loss_MSE = criterion_MSE(outputs, train_ind)
                loss_CE = criterion_CE(classification.squeeze(), labels)  # TODO: maybe add squeeze to class

                loss = (loss_MSE + loss_CE) / 2

                # backward pass
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()
                curr_loss += loss.item()
                curr_acc += self.accuracy(classification, labels)

            lr_scheduler.step()
            train_loss.append(curr_loss / len(self.train_loader))
            train_acc.append(curr_acc / len(self.train_loader))

            outputs_test, classification_test = model(test_ind)
            loss_MSE_test = criterion_MSE(outputs_test, test_ind)
            test_loss.append(loss_MSE_test.item())
            test_acc.append(self.accuracy(classification_test, test_labels))

        return train_loss, train_acc, test_loss, test_acc

    def reconstruct(self, data):
        return self.AE.to(self.device).forward(data.to(self.device))

    def accuracy(self, prob, labels):
        prediction = np.argmax(prob.squeeze().detach().cpu().numpy(), axis=1)
        return np.mean(prediction == labels.detach().cpu().numpy()) * 100

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

        amount_img = 3
        test_iter = iter(self.test_loader)
        test_images, test_labels = test_iter.next()
        test_images = test_images[:amount_img].squeeze()
        reconstruction = self.reconstruct(test_images).detach().cpu().squeeze().numpy()

        f, axs = plt.subplots(2, amount_img)

        for i in range(amount_img):
            axs[0, i].imshow(test_images[i], cmap='gray')
            axs[1, i].imshow(reconstruction[i], cmap='gray')

        axs[0, 0].set_ylabel("original")
        axs[1, 0].set_ylabel("constructed")
        plt.suptitle("Origin vs Reconstructed images")
        plt.show()

    def plot_classification(self, isRow):
        train_loss, train_acc, test_loss, test_acc = self.train_classification(isRow)
        x = np.arange(self.epochs)
        plt.plot(x, train_loss, label="Train loss")
        plt.plot(x, test_loss, label="Test loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(
            f'MNIST LSTM loss \n hidden state size = {self.hidden_state_sz}, learning rate = {self.lr}, gradient clipping = {self.grad_clip}')
        plt.show()

        plt.plot(x, train_loss, label="Train Accuracy")
        plt.plot(x, test_loss, label="Test Accuracy")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title(
            f'MNIST LSTM Accuracy \n hidden state size = {self.hidden_state_sz}, learning rate = {self.lr}, gradient clipping = {self.grad_clip}')
        plt.show()


ae = AE_MNIST(hidden_state_sz=hidden_state_sz, lr=lr, grad_clip=grad_clip)
ae.plot_classification(isRow=1)
