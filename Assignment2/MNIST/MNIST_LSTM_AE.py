import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_sz, hidden_state_sz, num_layers, dropout):
        super(Encoder, self).__init__()
        self.input_sz = input_sz
        self.hidden_state_sz = hidden_state_sz
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm_encoder = nn.LSTM(input_sz, hidden_state_sz, num_layers, batch_first=True, dropout=dropout)


    def forward(self, x):
        _, (ht, _) = self.lstm_encoder(x)
        return ht[self.num_layers - 1].view(-1, 1, self.hidden_state_sz)  #TODO: understand


class Decoder(nn.Module):
    def __init__(self, input_sz, hidden_state_sz, num_layers, dropout, seq_sz, output_sz):
        super(Decoder, self).__init__()
        self.input_sz = input_sz
        self.hidden_state_sz = hidden_state_sz
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_sz = seq_sz
        self.classification_sz = 10
        self.linear = nn.Linear(hidden_state_sz, output_sz)
        self.linear_classification = nn.Linear(hidden_state_sz, self.classification_sz)
        self.lstm_decoder = nn.LSTM(hidden_state_sz, hidden_state_sz, num_layers, batch_first=True, dropout=dropout)

    def forward(self, z):
        z = z.repeat(1, self.seq_sz, 1)
        output, (h_t, _) = self.lstm_decoder(z)
        return torch.tanh(self.linear(output)), F.softmax(self.linear_classification(h_t), dim=1)

class LSTM_AEC(nn.Module):
    def __init__(self, input_sz, hidden_state_sz, num_layers, dropout, seq_sz, output_sz):
        super(LSTM_AEC, self).__init__()
        self.input_sz = input_sz
        self.hidden_state_sz = hidden_state_sz
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_sz = seq_sz
        self.encoder = Encoder(input_sz, hidden_state_sz, num_layers, dropout)
        self.decoder = Decoder(input_sz, hidden_state_sz, num_layers, dropout, seq_sz, output_sz)


    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
