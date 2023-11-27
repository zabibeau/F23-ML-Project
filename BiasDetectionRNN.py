import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from torcheval.metrics.functional import multiclass_accuracy
from torcheval.metrics.functional import multiclass_precision
from torcheval.metrics.functional import multiclass_recall
from torcheval.metrics.functional import multiclass_f1_score

import sys

TEST_BATCH_SIZE = 512
training_accuracy = []

class RNN_embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim=25, hidden_size=128, num_layers=3, num_classes=3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :] # -1 only take the last time step
        out = self.fc(out)
        return out
    


def train(model, train_loader, n_epochs, optimizer, criterion, batch_size):
    model.train()

    for epoch in range(n_epochs):
        train_loss = 0.0
        accs = 0.0
        for i,(data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device).long()

            optimizer.zero_grad()
            predictions = model(data)
            preds = torch.argmax(predictions, dim=1)
            correct = (preds == label).sum().item()
            total = label.size(0)
            accuracy = correct / total
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            accs += accuracy
            if (i+1) % 100 == 0:
                print('Epoch: {} \tStep:{} \tTraining Loss: {:.6f} '.format(epoch+1, i+1, loss.item()))
        train_loss = train_loss/(len(train_loader.dataset)//batch_size)
        acc = accs/(len(train_loader.dataset) // batch_size)
        training_accuracy.append(acc)
        print(f'Epoch: {epoch + 1} \tTraining Loss: {train_loss: .6f}\t Accuracy: {acc: .6f}')



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

'''if len(sys.argv) != 5:
    print("Usage: python BiasDetectionRNN.py <batch_size> <hidden_size> <num_layers> <num_epochs>")
    exit(0)

batch_size = int(sys.argv[1])
hidden_size = int(sys.argv[2])
num_layers = int(sys.argv[3])
num_epochs = int(sys.argv[4])
#file.write(f"Params: Batch Size: {batch_size}\t Hidden Size: {hidden_size}\t Layers: {num_layers}\n")
train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_ds, shuffle=True, batch_size=TEST_BATCH_SIZE, drop_last=True)

model_embedding = RNN_embedding(len(word2id), hidden_size=hidden_size, num_layers=num_layers)
model_embedding = model_embedding.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_embedding.parameters(), lr = 1e-4)

train(model_embedding, train_loader, num_epochs, optimizer, criterion, batch_size)
evaluate(model_embedding, test_loader)'''
#file.write("=============================================================================\n")
#file.close()


