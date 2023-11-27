import BiasDetectionNN as bdnn
import BiasDetectionRNN as bdrnn

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchtext.data import get_tokenizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAINING_BATCH_SIZE = 64
TESTING_BATCH_SIZE = 512
NUM_EPOCHS = 5

training_accuracy = []
testing_accuracy = []
training_loss = []
testing_loss = []
iters = np.arange(1, NUM_EPOCHS + 1)

file = open("TestingOutput.txt", "wt")
# Getting training and testing dataframes

left = ["Atlantic", "Vox"]
leans_left = ["Buzzfeed News", "Business Insider", "CNN", "Guardian", "New York Times", "NPR", "Talking Points Memo", "Washington Post"]
neutral = ["Reuters"]
leans_right= ["National Review", "New York Post"]
right = ["Breitbart", "Fox News"]

data_files = ["data/articles1.csv", "data/articles2.csv", "data/articles3.csv"]
df = pd.concat((pd.read_csv(filename, usecols=["id", "title", "publication"]) for filename in data_files))

print("Dataframes loaded")

df.replace(left, value = 0, inplace=True)
df.replace(leans_left, value = 0, inplace=True)
df.replace(neutral, value = 1, inplace=True)
df.replace(leans_right, value = 2, inplace=True)
df.replace(right, value = 2, inplace=True)    

train_set, test_set = train_test_split(df, test_size = 0.15)

print("Datasets imported and ready to train from")

# Turns dataframes into lists
train_set = list(train_set.to_records(index=False))
test_set = list(test_set.to_records(index=False))

tokenizer = get_tokenizer("basic_english")

train_set = [(target, tokenizer(str(title))) for id, title, target in train_set]
test_set = [(target, tokenizer(str(title))) for id, title, target in test_set]

word_vocab = set()
word_vocab.add("<Start>")
word_vocab.add("<End>")
word_vocab.add("<Pad>")

for record in [train_set, test_set]:
    for target, title in record:
        for word in title:
            word_vocab.add(word)

word2id = {word: id for id, word in enumerate(word_vocab)}

def encode_and_pad(title, max_len):
    start = [word2id["<Start>"]]
    end = [word2id["<End>"]]
    pad = [word2id["<Pad>"]]

    if len(title) < max_len - 2: # 2 word for Start and End
        n_pads = max_len - 2 - len(title)
        encoded = [word2id[w] for w in title]
        return start + encoded + end + pad * n_pads
    else:
        encoded = [word2id[w] for w in title]
        truncated = encoded[:max_len - 2]
        return start + truncated + end
    
train_encoded = [(encode_and_pad(title, 25), target) for target, title in train_set]
test_encoded = [(encode_and_pad(title, 25), target) for target, title in test_set]

train_x = np.array([title for title, target in train_encoded])
train_y = np.array([target for title, target in train_encoded])
test_x = np.array([title for title, target in test_encoded])
test_y = np.array([target for title, target in test_encoded])

train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

train_loader = DataLoader(train_ds, shuffle=True, batch_size=TRAINING_BATCH_SIZE, drop_last=True)
test_loader = DataLoader(test_ds, shuffle=True, batch_size=TESTING_BATCH_SIZE, drop_last=True)

nn = bdnn.NeuralNetwork(len(word2id), embedding_dim=25, hidden_size=128).to(device)
rnn = bdrnn.RNN_embedding(len(word2id), hidden_size=128, num_layers=2).to(device)

criterion = torch.nn.CrossEntropyLoss()
nn_opt = torch.optim.Adam(nn.parameters(), lr = 1e-4)
rnn_opt = torch.optim.Adam(rnn.parameters(), lr = 1e-4)

bdnn.train(nn, train_loader, NUM_EPOCHS, nn_opt, criterion, TRAINING_BATCH_SIZE)
nn_acc = bdnn.training_accuracy
file.write("NN:\n")
file.write(bdnn.evaluate(nn, test_loader))

bdrnn.train(rnn, train_loader, NUM_EPOCHS, rnn_opt, criterion, TRAINING_BATCH_SIZE)
rnn_acc = bdrnn.training_accuracy
file.write("RNN:\n")
file.write(bdrnn.evaluate(rnn, test_loader))

file.close()

plt.plot(iters, nn_acc, label='MLP Accuracy')
plt.plot(iters, rnn_acc, label='RNN Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



