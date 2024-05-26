import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output.reshape(output.size(0)*output.size(1), output.size(2)))
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)

class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output.reshape(output.size(0)*output.size(1), output.size(2)))
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                torch.zeros(self.n_layers, batch_size, self.hidden_size))

if __name__ == '__main__':
    # Define parameters
    input_size = 65  # Number of unique characters
    hidden_size = 128
    output_size = 65
    n_layers = 2
    batch_size = 1

    # Test CharRNN
    rnn = CharRNN(input_size, hidden_size, output_size, n_layers)
    input_seq = torch.randint(0, input_size, (batch_size, 30))
    hidden = rnn.init_hidden(batch_size)
    output, hidden = rnn(input_seq, hidden)
    print("CharRNN output shape:", output.shape)

    # Test CharLSTM
    lstm = CharLSTM(input_size, hidden_size, output_size, n_layers)
    hidden = lstm.init_hidden(batch_size)
    output, hidden = lstm(input_seq, hidden)
    print("CharLSTM output shape:", output.shape)
