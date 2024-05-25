# Character-Level Language Modeling with RNN and LSTM

This project implements a character-level language model using vanilla RNN and LSTM networks. The model is trained on the Shakespeare dataset and is capable of generating text based on seed characters. This repository includes the following components:

1. `dataset.py`: A custom dataset class to load and preprocess the Shakespeare dataset.
2. `model.py`: Implementation of vanilla RNN and LSTM models.
3. `main.py`: Training script to train the models and monitor the training process.
4. `generate.py`: Script to generate text using the trained model.
5. `shakespeare.txt` : contains a collection of Shakespeare's works. You can download the dataset from [here](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) and save it as `shakespeare.txt` in the working directory.

## Requirements

1. **Data Pipeline**: Implemented in `dataset.py`, which loads the input file, constructs a character dictionary, and prepares data for training.
2. **Model Implementation**: Vanilla RNN and LSTM models are implemented in `model.py`. Stacked layers are used to enhance model performance.
3. **Training Script**: `main.py` includes the training loop and monitors average loss values for both training and validation datasets.
4. **Loss Plotting and Comparison**: The training and validation loss values are plotted to compare the performance of the vanilla RNN and LSTM models.
5. **Text Generation**: `generate.py` generates text using the trained model with various temperature settings.
6. **Softmax with Temperature**: The softmax function with a temperature parameter is implemented to generate more plausible text.

## Usage

### 1. Download and Prepare Dataset

Ensure you have the Shakespeare dataset (`shakespeare.txt`) in the working directory.

### 2. Train the Model

To train the models, run the following command:

```
python main.py
```
This script will train both the RNN and LSTM models and save the model with the best validation performance.

### 3. Generate Text
After training, use the generate.py script to generate text:

```
python generate.py
```
This script will generate text samples of length 100 from different seed characters using the best-performing model.

## Example Results
### Training and Validation Loss

### Generated Text Samples
**Temperature: 0.5**

```
Seed: T
Generated Sample: Th and and and the and the and the and the and the and the and the and the and the and
```

**Temperature: 1.0**
```
Seed: H
Generated Sample: How all the sighted and sease and with my self, the hast or the tale,
That we see the time for my father's dead,
```

**Temperature: 1.5**

```
Seed: E
Generated Sample: Execithing,
For so; still trether. I be to brow her fords,
Whin! It be ho; my him youl;
A father:
```

### Discussion on Temperature
Low Temperature (e.g., 0.5): Generates repetitive and predictable text. The model is more conservative, producing less diverse outputs.
Medium Temperature (e.g., 1.0): Balances between diversity and coherence, producing more natural and plausible text.
High Temperature (e.g., 1.5): Generates more diverse and creative text but may become less coherent and produce more errors.

## Conclusion
This project demonstrates the effectiveness of character-level language models using RNN and LSTM networks. By experimenting with different model architectures and temperature settings, we can generate diverse and coherent text.

## References
Karpathy, A. (2015). The Unreasonable Effectiveness of Recurrent Neural Networks. Link
