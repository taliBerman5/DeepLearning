import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from Assignment2.SNP500 import SNP_data
import torch
from Assignment2.LSTM_AE import LSTM_AE as AE


batch = 8
epochs = 3
optimizer = torch.optim.Adam
hidden_state_sz = 27
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
        self.output_sz = output_sz
        self.grad_clip = grad_clip
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.AE = AE(self.input_sz, self.hidden_state_sz, self.num_layers, self.dropout, self.seq_sz,
                     self.output_sz)
        self.optimizer = optimizer(self.AE.parameters(), lr=self.lr)



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

    def reconstruct(self, data):
        data = torch.flatten(data, 0, 1).unsqueeze(2).to(self.device)
        return self.AE.to(self.device).forward(data.to(self.device))

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
        test_iter = iter(self.test_loader)
        test_images = test_iter.next()
        test_images = test_images[:amount_img].squeeze()
        reconstruction = self.reconstruct(test_images).detach().cpu().squeeze().numpy()

        test_images = test_images.detach().cpu().squeeze().numpy()
        test_images = SNP_data.revert_normalize(test_images, data_dict['test_mean'], data_dict['test_std'])
        reconstruction = SNP_data.revert_normalize(reconstruction, data_dict['test_mean'], data_dict['test_std'])

        f, axs = plt.subplots(2, amount_img)

        for i in range(amount_img):
            axs[0, i].imshow(test_images[i], cmap='gray')
            axs[1, i].imshow(reconstruction[i], cmap='gray')

        axs[0, 0].set_ylabel("original")
        axs[1, 0].set_ylabel("constructed")
        plt.suptitle("Origin vs Reconstructed images")
        plt.show()




# daily_stock_AMZN_GOOGL()
model = AE_SNP500()
model.plot()
