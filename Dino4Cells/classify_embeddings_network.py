import argparse
from MLkNN import MLkNN
from multiprocessing.pool import Pool
from faiss_knn import FaissKNeighbors, FaissMLkNN
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
# import faiss
from sklearn.linear_model import LogisticRegression
from types import SimpleNamespace

parser = argparse.ArgumentParser('Visualize Self-Attention maps')
parser.add_argument('-f', type=str)
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
parser.add_argument('--train', type=str, help='Path to training set')
parser.add_argument('--train_other', type=str, default=None, help='Path to training set')
parser.add_argument('--test', type=str, help='Path to testing set')
parser.add_argument('--test_other', type=str, default=None, help='Path to testing set')
parser.add_argument('--classifier_path', type=str, help='Path to classifier')
parser.add_argument('-p', '--pretrained', action='store_true', help='Path to classifier')
parser.add_argument('--classifier_type', type=str, help='type of classifier to run')
parser.add_argument('--prediction_path', type=str, help='path to prediction')
args = parser.parse_args()


class Multilabel_classifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

X_train, y_train = np.load(args.train, allow_pickle=True)
X_test, y_test = np.load(args.test, allow_pickle=True)
if (type(args.train_other) == str) & (type(args.test_other) == str):
    print('combining features')
    X_train_other, y_train, _ = np.load(args.train_other, allow_pickle=True)
    X_test_other, y_test, _ = np.load(args.test_other, allow_pickle=True)
    X_train = torch.cat((X_train, X_train_other), axis=1)
    X_test = torch.cat((X_test, X_test_other), axis=1)

target_matrix_train = np.zeros((len(y_train), 28))
for i,c in enumerate(y_train):
    for class_ind in c.split():
        target_matrix_train[i, int(class_ind)] = 1

y_train = target_matrix_train

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

criterion = nn.BCELoss()

model = Multilabel_classifier(X_train.shape[1], y_train.shape[1])
optimizer = optim.SGD(model.parameters(), lr=0.01)
batch_size = 1000
num_epochs = 100
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
batches = [(X_train[i : i + batch_size, :], y_train[i : i + batch_size, :])
           for i in range(0, X_train.shape[0], batch_size)]


print(len(batches))
print(batches[0][0].shape)
print(batches[-1][0].shape)
pbar = tqdm(range(num_epochs))
running_loss = 0
for epoch in pbar:
    temp_running_loss = 0
    for ind, (X_batch, y_batch) in enumerate(batches):
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        temp_running_loss += loss.item()
        optimizer.step()
        pbar.set_description(f'epoch: {epoch}, batch: {ind}, loss: {loss.item():.2f}, running loss {running_loss:.2f}')
    running_loss = temp_running_loss / len(batches)
    torch.save(model.state_dict(), 'classifier.pth')

