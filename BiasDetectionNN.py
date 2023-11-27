import numpy as np
import pandas as pd
import sys
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as functional
import torch.optim as optim
from torchtext.data import get_tokenizer

from sklearn.model_selection import train_test_split

from torcheval.metrics.functional import multiclass_accuracy
from torcheval.metrics.functional import multiclass_precision
from torcheval.metrics.functional import multiclass_recall
from torcheval.metrics.functional import multiclass_f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAINING_BATCH_SIZE = 64
TESTING_BATCH_SIZE = 512
HIDDEN_LAYER_SIZE1 = 128
NUM_EPOCHS = 50
training_accuracy = []
testing_accuracy = []
training_loss = []
testing_loss = []
iters = np.arange(1, NUM_EPOCHS + 1)

#Class for our neural network
class NeuralNetwork(nn.Module):

    def __init__(self, vocab_size, embedding_dim=25, hidden_size=128, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden = nn.Linear(in_features=embedding_dim, out_features=hidden_size, bias=True)
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        out = functional.relu(self.hidden(x))
        #out = functional.relu(self.hidden(x))
        out = self.out(out)
        return out
    
    

# Training for model
def train(model, train_loader, n_epochs, optimizer, criterion, train_batch_size):
    model.train()
    
    for epoch in range(n_epochs):
        train_loss = 0.0
        accs = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).long()
            optimizer.zero_grad()

            predictions = model(inputs)
            preds = torch.argmax(predictions, dim=1)
            correct = (preds == labels).sum().item()
            total = labels.size(0)
            accuracy = correct / total
            loss = criterion(predictions, labels)
            #print(f"Outputs: {outputs}\t Labels: {labels}\t Loss: {loss}")
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            accs += accuracy
        train_loss = train_loss / (len(train_loader.dataset)//train_batch_size)
        acc = accs / (len(train_loader.dataset) // train_batch_size)
        training_accuracy.append(acc)
        training_loss.append(train_loss)
        #test_loss = getTestLoss(test_loader, model, criterion, test_batch_size)
        #testing_loss.append(test_loss)


        print(f"Epoch: {epoch + 1} \t Training Loss: {train_loss: .6f} \t Accuracy: {acc: .6f}")
    print("Training Complete")

def find_max_index(preds):
    max = 0
    res = []
    for pred in preds:
        for i in range(0, len(pred)):
            if pred[i] > pred[max]:
                max = i
        res.append(max)
    return res

def evaluate(model, test_loader):
    batch_acc = []
    batch_pre = []
    batch_recall = []
    batch_f1 = []
    for i,(data, label) in enumerate(test_loader):
        data, label = data.to(device), label.to(device)

        predictions = model(data)
        preds = torch.argmax(predictions, dim=1)
        acc = multiclass_accuracy(preds, label, num_classes=3)
        pre = multiclass_precision(preds, label, num_classes=3)
        recall = multiclass_recall(preds, label, num_classes=3)
        f1 = multiclass_f1_score(preds, label, num_classes=3)

        batch_acc.append(acc)
        batch_pre.append(pre)
        batch_recall.append(recall)
        batch_f1.append(f1)
    return 'Precision: {:.3f}\tRecall: {:.3f}\tF1:{:.3f}\tAccuracy: {:.3f}\n'.\
            format(sum(batch_pre)/len(batch_pre), sum(batch_recall)/len(batch_recall),
                    sum(batch_f1)/len(batch_f1), sum(batch_acc)/len(batch_acc))
    
#Uncomment for scripting

'''if len(sys.argv) != 4:
    print("Usage: python BiasDetectionNN.py <batch_size> <hidden_size> <num_epochs>")
    exit(0)

batch_size = int(sys.argv[1])
hidden_size = int(sys.argv[2])
num_epochs = int(sys.argv[3])
#file.write(f"Params: Batch Size: {batch_size}\t Hidden Size: {hidden_size}\t Layers: {num_layers}\n")
train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_ds, shuffle=True, batch_size=1000, drop_last=True)

model_embedding = NeuralNetwork(len(word2id), hidden_size=hidden_size)
model_embedding = model_embedding.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_embedding.parameters(), lr = 1e-4)

train(model_embedding, train_loader, num_epochs, optimizer, criterion, batch_size)
evaluate(model_embedding, test_loader)'''