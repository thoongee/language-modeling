import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from dataset import Shakespeare
from model import CharRNN, CharLSTM

def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    total_loss = 0
    for inputs, targets in trn_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        hidden = model.init_hidden(inputs.size(0))
        if isinstance(hidden, tuple):  # LSTM
            hidden = (hidden[0].to(device), hidden[1].to(device))
        else:  # RNN
            hidden = hidden.to(device)
        
        optimizer.zero_grad()
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs, targets.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    trn_loss = total_loss / len(trn_loader)
    return trn_loss

def validate(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = model.init_hidden(inputs.size(0))
            if isinstance(hidden, tuple):  # LSTM
                hidden = (hidden[0].to(device), hidden[1].to(device))
            else:  # RNN
                hidden = hidden.to(device)
            
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets.view(-1))
            total_loss += loss.item()
    val_loss = total_loss / len(val_loader)
    return val_loss

def main():
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    # Parameters
    input_file = 'shakespeare.txt'
    batch_size = 64
    seq_length = 30
    hidden_size = 128
    n_layers = 2
    n_epochs = 20
    learning_rate = 0.001
    validation_split = 0.2
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = Shakespeare(input_file)
    
    # Create train and validation splits
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    
    # Initialize models, loss function, and optimizer
    input_size = len(dataset.char_to_idx)
    output_size = input_size
    
    rnn_model = CharRNN(input_size, hidden_size, output_size, n_layers).to(device)
    lstm_model = CharLSTM(input_size, hidden_size, output_size, n_layers).to(device)
    
    criterion = nn.CrossEntropyLoss()
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
    
    # Variables to track best model
    best_val_loss = float('inf')
    best_model = None
    best_model_type = None
    
    # Training loop
    rnn_train_losses, rnn_val_losses = [], []
    lstm_train_losses, lstm_val_losses = [], []
    
    for epoch in range(n_epochs):
        rnn_train_loss = train(rnn_model, train_loader, device, criterion, rnn_optimizer)
        rnn_val_loss = validate(rnn_model, val_loader, device, criterion)
        
        lstm_train_loss = train(lstm_model, train_loader, device, criterion, lstm_optimizer)
        lstm_val_loss = validate(lstm_model, val_loader, device, criterion)
        
        rnn_train_losses.append(rnn_train_loss)
        rnn_val_losses.append(rnn_val_loss)
        lstm_train_losses.append(lstm_train_loss)
        lstm_val_losses.append(lstm_val_loss)
        
        print(f'Epoch {epoch+1}/{n_epochs}, RNN Train Loss: {rnn_train_loss:.4f}, RNN Val Loss: {rnn_val_loss:.4f}, LSTM Train Loss: {lstm_train_loss:.4f}, LSTM Val Loss: {lstm_val_loss:.4f}')
        
        # Save the best model
        if rnn_val_loss < best_val_loss:
            best_val_loss = rnn_val_loss
            best_model = rnn_model.state_dict()
            best_model_type = 'RNN'
        if lstm_val_loss < best_val_loss:
            best_val_loss = lstm_val_loss
            best_model = lstm_model.state_dict()
            best_model_type = 'LSTM'
    
    # Save the best model to disk
    torch.save(best_model, f'best_model_{best_model_type}.pth')
    
    # Plotting the loss curves
    plt.figure()
    plt.plot(rnn_train_losses, label='RNN Train Loss')
    plt.plot(rnn_val_losses, label='RNN Val Loss')
    plt.plot(lstm_train_losses, label='LSTM Train Loss')
    plt.plot(lstm_val_losses, label='LSTM Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.show()

if __name__ == '__main__':
    main()
