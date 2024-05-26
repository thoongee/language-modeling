import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
           You need this dictionary to generate characters.
        2) Make list of character indices using the dictionary
        3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file):
        # Load input file
        with open(input_file, 'r') as f:
            self.text = f.read()
        
        # Create character dictionary
        self.chars = sorted(list(set(self.text)))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}

        # Convert text to indices
        self.data = [self.char_to_idx[char] for char in self.text]
        
        # Split data into sequences of length 30
        self.seq_length = 30
        self.sequences = []
        self.targets = []
        
        for i in range(0, len(self.data) - self.seq_length):
            self.sequences.append(self.data[i:i + self.seq_length])
            self.targets.append(self.data[i + 1:i + self.seq_length + 1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        target_seq = torch.tensor(self.targets[idx], dtype=torch.long)
        return input_seq, target_seq

if __name__ == '__main__':
    # Test code to verify your implementation
    dataset = Shakespeare('shakespeare.txt')
    print(f'Dataset size: {len(dataset)}')
    input_seq, target_seq = dataset[0]
    print(f'Input sequence: {input_seq}')
    print(f'Target sequence: {target_seq}')
