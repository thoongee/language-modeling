import matplotlib.pyplot as plt

# Given data
epochs = list(range(1, 21))
rnn_train_losses = [
    1.5542, 1.4496, 1.4323, 1.4236, 1.4183, 1.4146, 1.4118, 1.4097, 1.4081, 
    1.4067, 1.4055, 1.4046, 1.4038, 1.4032, 1.4024, 1.4019, 1.4014, 1.4010, 
    1.4006, 1.4003
]
rnn_val_losses = [
    1.4665, 1.4407, 1.4296, 1.4237, 1.4187, 1.4174, 1.4136, 1.4114, 1.4089, 
    1.4084, 1.4074, 1.4087, 1.4059, 1.4054, 1.4034, 1.4044, 1.4015, 1.4028, 
    1.4039, 1.4042
]
lstm_train_losses = [
    1.5131, 1.3536, 1.3154, 1.2929, 1.2773, 1.2660, 1.2571, 1.2502, 1.2444, 
    1.2396, 1.2356, 1.2322, 1.2293, 1.2268, 1.2245, 1.2226, 1.2209, 1.2192, 
    1.2178, 1.2165
]
lstm_val_losses = [
    1.3865, 1.3329, 1.3057, 1.2892, 1.2774, 1.2674, 1.2601, 1.2551, 1.2495, 
    1.2479, 1.2436, 1.2405, 1.2378, 1.2363, 1.2336, 1.2314, 1.2291, 1.2285, 
    1.2289, 1.2296
]

# Plotting the losses
plt.figure(figsize=(10, 6))
plt.plot(epochs, rnn_train_losses, label='RNN Train Loss')
plt.plot(epochs, rnn_val_losses, label='RNN Val Loss')
plt.plot(epochs, lstm_train_losses, label='LSTM Train Loss')
plt.plot(epochs, lstm_val_losses, label='LSTM Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses for RNN and LSTM')
plt.legend()
plt.grid(True)
plt.show()
