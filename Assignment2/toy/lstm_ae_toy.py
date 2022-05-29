import torch
from synthetic_data import toy_data
from Assignment2.LSTM_AE import LSTM_AE as AE
import numpy as np
import math
import matplotlib.pyplot as plt

batch = 100
epochs = 20
optimizer = torch.optim.Adam
hidden_state_sz = 100
num_layers = 1
lr = 0.01
input_sz = 1
dropout = 0
seq_sz = 50
output_sz = 1
grad_clip = None


class AE_TOY():
    def __init__(self, train, validation, test, hidden_state_sz, lr, grad_clip):
        self.train_data = train
        self.validation_data = validation
        self.test_data = test
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
        self.lstm = AE(self.input_sz, self.hidden_state_sz, self.num_layers, self.dropout, self.seq_sz, self.output_sz)
        self.optimizer = optimizer(self.lstm.parameters(), lr=self.lr)

    def train(self):
        amount_data = self.train_data.size(dim=0)
        model = self.lstm.to(self.device)
        criterion = torch.nn.MSELoss().to(self.device)
        stepper = torch.optim.lr_scheduler.StepLR(self.optimizer, 100, 0.5)

        train_loss = []
        validation_loss = []

        for epoch in range(self.epochs):
            print(f'stated epoche {epoch}')
            rnd_ind = np.random.permutation(amount_data)

            curr_loss = 0

            for b in range(math.floor(amount_data / self.batch)):
                ind = rnd_ind[b * batch: (b + 1) * batch]
                train_ind = self.train_data[ind, :, :].to(self.device)
                self.optimizer.zero_grad()

                # forward pass
                outputs = model(train_ind)
                loss = criterion(outputs, train_ind)

                # backword pass
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                self.optimizer.step()
                curr_loss += loss.item()

            stepper.step()
            train_loss.append(curr_loss / math.floor(amount_data / self.batch))

            v_data = self.validation_data.to(self.device)
            outputs = model(v_data)
            validation_loss.append(criterion(outputs, v_data).item())

        return train_loss, validation_loss

    def plot(self):
        train_loss, validation_loss = self.train()
        x = np.arange(self.epochs)
        plt.plot(x, train_loss, label="Train loss")
        plt.plot(x, validation_loss, label="validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Toy LSTM loss")
        plt.show()


train, validation, test = toy_data()
ae = AE_TOY(train, validation, test, hidden_state_sz=100, lr=0.01, grad_clip=200)

ae.plot()
