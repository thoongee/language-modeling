import torch
import torch.nn.functional as F
from model import CharRNN, CharLSTM
import dataset

def generate(model, seed_characters, temperature, char_to_idx, idx_to_char, device, length=100):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
        temperature: T
        char_to_idx: character to index dictionary
        idx_to_char: index to character dictionary
        device: device for computing, cpu or gpu
        length: length of the generated sequence

    Returns:
        samples: generated characters
    """
    model.eval()
    hidden = model.init_hidden(1)
    if isinstance(hidden, tuple):  # LSTM
        hidden = (hidden[0].to(device), hidden[1].to(device))
    else:  # RNN
        hidden = hidden.to(device)
    input_seq = torch.tensor([char_to_idx[c] for c in seed_characters], dtype=torch.long).unsqueeze(0).to(device)
    
    samples = seed_characters
    
    for _ in range(length):
        output, hidden = model(input_seq, hidden)
        output = output / temperature
        probabilities = F.softmax(output, dim=1).data
        char_idx = torch.multinomial(probabilities[-1], 1).item()  # Select the last time step's probabilities
        char = idx_to_char[char_idx]
        samples += char
        input_seq = torch.tensor([[char_idx]], dtype=torch.long).to(device)
    
    return samples

def main():
    # Parameters
    input_file = 'shakespeare.txt'
    hidden_size = 128
    n_layers = 2
    model_paths = {'RNN': 'best_model_RNN.pth', 'LSTM': 'best_model_LSTM.pth'}
    temperature = [0.5, 1.0, 1.5]
    seed_characters = ['T', 'H', 'E', 'R', 'S']
    length = 100
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    shakespeare_dataset = dataset.Shakespeare(input_file)
    char_to_idx = shakespeare_dataset.char_to_idx
    idx_to_char = shakespeare_dataset.idx_to_char
    input_size = len(char_to_idx)
    output_size = input_size
    
    # Determine the model type to load
    model_type = 'LSTM'  # or 'RNN', based on the best model type from main.py
    model_class = CharLSTM if model_type == 'LSTM' else CharRNN
    model = model_class(input_size, hidden_size, output_size, n_layers).to(device)
    model.load_state_dict(torch.load(model_paths[model_type]))
    
    # Generate and print samples for different temperatures
    for temp in temperature:
        print(f'Temperature: {temp}')
        for seed in seed_characters:
            sample = generate(model, seed, temp, char_to_idx, idx_to_char, device, length)
            print(f'Seed: {seed}\nGenerated Sample: {sample}\n')

if __name__ == '__main__':
    main()
