# Character-Level Language Modeling with RNN and LSTM

This project implements a character-level language model using the Shakespeare dataset, using RNN and LSTM networks that can generate text based on seed characters. 
By experimenting with different model architectures and temperature settings, demonstrates how to generate diverse and coherent text.

## Files
1. `dataset.py`: A custom dataset class to load and preprocess the Shakespeare dataset.
2. `model.py`: Implementation of vanilla RNN and LSTM models.
3. `main.py`: Training script to train the models and monitor the training process.
4. `generate.py`: Script to generate text using the trained model.
5. `shakespeare.txt` : contains a collection of Shakespeare's works. You can download the dataset from [here](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) and save it as `shakespeare.txt` in the working directory.

## Usage

### 1. Download and Prepare Dataset

Ensure you have the Shakespeare dataset (`shakespeare.txt`) in the working directory.

### 2. Train the Model

To train the models, run the following command:

```
python3 main.py
```
This script will train both the RNN and LSTM models and save the model with the best validation performance.

### 3. Generate Text
After training, use the generate.py script to generate text:

```
python3 generate.py
```
This script will generate text samples of length 100 from different seed characters using the best-performing model.

## Example Results
### 1. Training and Validation Loss

<p align="center"><img src="https://github.com/thoongee/language-modeling/assets/94193480/3ace24bf-4cd7-451a-9182-9eca9587ddc7" width="50%">

**RNN:**
- The training loss and validation loss gradually decrease.
- The validation loss is higher than the training loss, indicating that the model is less accurate on the validation data compared to the training data.
  
**LSTM:**
- Both the training loss and validation loss of the LSTM continuously decrease.
- Overall, the LSTM shows lower loss values compared to the RNN, indicating that the LSTM performs better.

In conclusion, the LSTM model demonstrates better language generation performance compared to the RNN model. LSTMs are particularly effective in handling long-term dependencies over time, making them more suitable for text generation tasks.

### 2. Generated Text Samples

**Seed:**

- Seed characters are the initial characters used to start text generation. The model predicts the next character based on these seed characters and repeats the process to generate text of the specified length.
Example: T, H, E, R, S

**Temperature:**

- The temperature parameter is applied to the softmax function to adjust the output probability distribution of the model.
Example: 0.5, 1.0, 1.5

**Results Interpretation**

Temperature: 0.5:
```
Seed: T: "The needless son to see thee for the faultless heart, To be a bloody conscience in his company. GREE"
Seed: H: "Hastiffess to my son, and made the state the world as your good angel, That thou wilt have a proud no"
Seed: E: "E: Why, no, no, not be the castle make a brother, And so in hands of his lady to the son of such a th"
Seed: R: "RENCE: There will not see the state with out the happy days Which we have a standing with shall have"
Seed: S: "S: Why, what a state and happy days for an absency are fled To stay the time of one of the season of"
```
- With a low temperature value (0.5), the generated text is consistent, repetitive, and less creative. The model operates more conservatively, selecting the most probable outputs. As a result, the context is well-maintained, but diversity may be lacking.

Temperature: 1.0:

```
Seed: T: "THIO: Nb: Master Francating, yet little Commise, my dearest Emmanisment, Should better once, forth,"
Seed: H: "He's kind of alterate of precious city Margaret, My mirth excelling. MENENIUS: Ay, proud sake! JULI"
Seed: E: "Edward's other; Beseech you, while together. Wen, what a tear, a most have; And, yet if the ordering"
Seed: R: "RD IV: Have it that doth head, Her anceit my son Coriolanus; but well with sance had not, God he can"
Seed: S: "S: No more of thy footes for his neighbour: Here come the duke they have with a gentleman of such a f"
```

- At medium temperature (1.0), the generated text strikes a balance between consistency and variety. The model behaves somewhat stochastically, generating a wide variety of output, but still maintains good context.

Temperature: 1.5:
```
Seed: T: "TIA: Away, I know but ourse one boy: Op thy flesh within, revenge-day. NORTHUMBERLAND: Sick-bloody-u"
Seed: H: "Haste: let devils and pock'd high our policy! A flight, no fall for Boshel'st! VAUGHALIN: It would g"
Seed: E: "EOTUS: 'Tis pale, As amisto camoroughorance jewel throw the imman! Here,--speak! Cuckile--rid you. Yo"
Seed: R: "RUCHIO: Nay, believe the vaposfedics.' Cormurath, scearful Landener. Peter, warriors? Bethindio, more"
Seed: S: "S: Plant way! THOMAS MOWBRAY: Brocketion, giver, I have knave Tilte intends, sir; I can door, and th"
```
- With a high temperature value (1.5), the generated text is very diverse and creative. The model operates more probabilistically, producing unpredictable outputs. This allows for the generation of unique text, but context consistency may be compromised.

**Discussion**
- Seed characters play a crucial role in starting text generation. Using different seed characters can generate diverse texts from different starting points.
- The temperature parameter greatly affects the variety and consistency of the generated text: lower temperatures produce more conservative and consistent text, while higher temperatures produce more creative and varied text. Medium temperatures strike a balance between variety and consistency, and are most effective at generating text that feels natural and plausible. Therefore, a temperature of 1.0 is effective at producing the most plausible results.


## References
- [Karpathy, A. (2015). The Unreasonable Effectiveness of Recurrent Neural Networks.](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
