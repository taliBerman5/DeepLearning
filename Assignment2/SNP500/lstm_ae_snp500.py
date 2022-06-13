import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from Assignment2.SNP500 import SNP_data
import torch
from Assignment2.LSTM_AE import LSTM_AE as AE
from Assignment2.SNP500.SNP_LSTM_AE import LSTM_AEP as AEP

batch = 8
epochs = 50
optimizer = torch.optim.Adam
hidden_state_sz = 50
num_layers = 1
lr = 0.001
input_sz = 1
dropout = 0
seq_sz = 53
output_sz = 1
grad_clip = 1

data_dict = SNP_data.split_data(SNP_data.parse())
train_loader = torch.utils.data.DataLoader(data_dict['train_set'], batch_size=batch, shuffle=True)
test_loader = torch.utils.data.DataLoader(data_dict['test_set'], batch_size=len(data_dict['test_set']), shuffle=False)


def daily_stock_AMZN_GOOGL():
    stocks = SNP_data.parse()
    AMZN_stocks = stocks[stocks['symbol'] == 'AMZN'][['date', 'high']]
    GOOGL_stocks = stocks[stocks['symbol'] == 'GOOGL'][['date', 'high']]

    fig, axes = plt.subplots()
    axes.xaxis.set_major_locator(MaxNLocator(5))
    plt.plot(AMZN_stocks['date'], AMZN_stocks['high'], label="AMZN stock")
    plt.plot(GOOGL_stocks['date'], GOOGL_stocks['high'], label="GOOGL stock")
    plt.title("GOOGL vs AMZN stock")
    plt.legend()
    plt.show()


class AE_SNP500():
    def __init__(self):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.batch = batch
        self.epochs = epochs
        self.hidden_state_sz = hidden_state_sz
        self.input_sz = input_sz
        self.seq_sz = seq_sz
        self.seq_predict_sz = seq_sz - 1
        self.output_sz = output_sz
        self.grad_clip = grad_clip
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.AE = AE(self.input_sz, self.hidden_state_sz, self.num_layers, self.dropout, self.seq_sz,
                     self.output_sz)
        self.optimizer = optimizer(self.AE.parameters(), lr=self.lr)
        self.AEP = AEP(self.input_sz, self.hidden_state_sz, self.num_layers, self.dropout, self.seq_predict_sz,
                       self.output_sz)
        self.optimizer_predict = optimizer(self.AEP.parameters(), lr=self.lr)

    def train(self):
        model = self.AE.to(self.device)
        criterion = torch.nn.MSELoss().to(self.device)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 100, 0.5)

        train_loss = []

        for epoch in range(self.epochs):
            print(f'started epoch {epoch}')

            curr_loss = 0
            for b, train_ind in enumerate(self.train_loader):
                train_ind = torch.flatten(train_ind, 0, 1).unsqueeze(2).to(self.device)
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

            lr_scheduler.step()
            train_loss.append(curr_loss / len(self.train_loader))

        return train_loss

    def train_predict(self):
        model = self.AEP.to(self.device)
        criterion = torch.nn.MSELoss().to(self.device)

        total_loss = []
        reconstruct_loss = []
        predict_loss = []

        for epoch in range(self.epochs):
            print(f'started epoch {epoch}')

            curr_total_loss = 0
            curr_reconstruct_loss = 0
            curr_predict_loss = 0
            for b, train_ind in enumerate(self.train_loader):
                train_ind = torch.flatten(train_ind, 0, 1).unsqueeze(2)
                X = train_ind[:, :-1].to(self.device)
                Y = train_ind[:, 1:].to(self.device)
                self.optimizer_predict.zero_grad()

                # forward pass
                outputs, pred = model(X)
                loss_reconstruct = criterion(outputs, X)
                loss_predict = criterion(pred, Y.squeeze())

                loss = (loss_reconstruct + loss_predict) / 2

                # backward pass
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                self.optimizer_predict.step()
                curr_total_loss += loss.item()
                curr_reconstruct_loss += loss_reconstruct.item()
                curr_predict_loss += loss_predict.item()

            divider = len(self.train_loader)
            total_loss.append(curr_total_loss / divider)
            reconstruct_loss.append(curr_reconstruct_loss / divider)
            predict_loss.append(curr_predict_loss / divider)

        return total_loss, reconstruct_loss, predict_loss


    def reconstruct(self, data):
        data = torch.flatten(data, 0, 1).unsqueeze(2).to(self.device)
        reconstruct = self.AE.to(self.device).forward(data.to(self.device))
        return reconstruct.view(-1, 19, 53)

    def reconstruct_predict(self, data):
        data = torch.flatten(data, 0, 1).unsqueeze(2).to(self.device)
        data = data[:, :-1].to(self.device)
        reconstruct, predict = self.AEP.to(self.device).forward(data.to(self.device))
        return predict.view(-1, 19, 52)

    def plot(self):
        train_loss = self.train()
        x = np.arange(self.epochs)
        plt.plot(x, train_loss, label="Train loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(
            f'S&P 500 LSTM loss \n hidden state size = {self.hidden_state_sz}, learning rate = {self.lr}, gradient clipping = {self.grad_clip}')
        plt.show()

        amount_img = 3
        test_images, reconstruction = self.get_reconstruct_and_test(amount_img, self.reconstruct)
        test_images, reconstruction = self.revert_normalize_data(test_images, reconstruction)

        for i in range(amount_img):
            fig, axes = plt.subplots()
            axes.xaxis.set_major_locator(MaxNLocator(5))
            plt.plot(data_dict['dates'], test_images[i].flatten(), label='original')
            plt.plot(data_dict['dates'], reconstruction[i].flatten(), label='reconstructed')
            plt.title(f"Original vs Reconstructed, symbol={data_dict['test_name'][i][0]}")
            plt.xlabel("Date")
            plt.ylabel("High Rate")
            plt.legend()
            plt.show()

    def plot_predict(self):
        total_loss, reconstruct_loss, predict_loss = self.train_predict()
        x = np.arange(self.epochs)
        plt.plot(x, reconstruct_loss, label="Train reconstruct loss")
        plt.plot(x, predict_loss, label="Train predict loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(
            f'S&P 500 reconstruct loss vs predict loss \n hidden state size = {self.hidden_state_sz}, learning rate = {self.lr}, gradient clipping = {self.grad_clip}')
        plt.show()
        amount_img = 2
        test_images, reconstruction = self.get_reconstruct_and_test(amount_img, self.reconstruct_predict)

        test_images = torch.flatten(test_images, 0, 1).unsqueeze(2)
        y = test_images[:, 1:].view(-1, 19, 52)
        y, reconstruction = self.revert_normalize_data(y, reconstruction)
        dates = data_dict['dates'].reshape(53, 19)[:-1].flatten()

        for i in range(amount_img):
            fig, axes = plt.subplots()
            axes.xaxis.set_major_locator(MaxNLocator(5))
            plt.plot(dates, y[i].flatten(), label='original')
            plt.plot(dates, reconstruction[i].flatten(), label='predicted')
            plt.title(f"Original vs predicted one step, symbol={data_dict['test_name'][i][0]}")
            plt.xlabel("Date")
            plt.ylabel("High Rate")
            plt.legend()
            plt.show()



    def get_reconstruct_and_test(self, amount_img, reconstruct_func):
        test_iter = iter(self.test_loader)
        test_images = test_iter.next()
        test_images = test_images[:amount_img].squeeze()
        reconstruction = reconstruct_func(test_images).detach().cpu().squeeze().numpy()
        return test_images, reconstruction

    def revert_normalize_data(self, test_images, reconstruction):
        test_images = test_images.detach().cpu().squeeze().numpy()
        SNP_data.revert_normalize(test_images, data_dict['test_mean'], data_dict['test_std'])
        SNP_data.revert_normalize(reconstruction, data_dict['test_mean'], data_dict['test_std'])

        return test_images, reconstruction


# daily_stock_AMZN_GOOGL()
model = AE_SNP500()
model.plot_predict()
