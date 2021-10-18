import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.linear_model import LogisticRegression
from types import SimpleNamespace
from label_dict import protein_to_num_full, protein_to_num_5k

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import yaml

parser = argparse.ArgumentParser('Get embeddings from model')
parser.add_argument('--config', type=str, default='.', help='path to config file')

args = parser.parse_args()
config = yaml.safe_load(open(args.config, 'r'))

class MLLR:
    def __init__(self, max_iter=100, num_classes = None):
        self.index = None
        self.y = None
        self.num_classes = num_classes
        self.max_iter = max_iter

    def fit(self, X, y):
        if type(self.num_classes) == type(None):
            self.num_classes = np.array(y).shape[1]

        self.classifiers = [LogisticRegression(max_iter = self.max_iter) for c in range(self.num_classes)]

        pbar = tqdm(enumerate(self.classifiers))
        for ind, c in pbar:
            pbar.set_description(f'training LinearRegressor for class {ind}')
            c.fit(X,y[:,ind])

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.num_classes))
        for ind, c in enumerate(self.classifiers):
            predictions[:,ind] = c.predict(X)
        return predictions

train_id_list = np.genfromtxt(config['classification']['train_inds'], dtype=str, delimiter=',')
test_id_list = np.genfromtxt(config['classification']['test_inds'], dtype=str, delimiter=',')
files = pd.read_csv(config['embedding']['df_path'])
files = files[~(files.file.apply(lambda impath: ('30871' in impath) or ('27093' in impath) or ('35134' in impath)))]

features_list = config['classification']['features_paths_list']
all_features, protein_locations, cell_types = torch.load(features_list[0])
if len(features_list) > 1:
    for features_path in features_list[1:]:
        all_features_other, _, _ = torch.load(features_path)
        all_features = torch.cat((all_features, all_features_other), axis=1)

ids = (files.ID).iloc[:len(all_features)]
train_id_list = list(set(list(train_id_list)) & set(list(ids)))
test_id_list = list(set(list(test_id_list)) & set(list(ids)))
train_inds = np.where((ids.isin(train_id_list)))[0]
test_inds = np.where((ids.isin(test_id_list)))[0]

cell_types = files.cell_type
cell_types = np.array(cell_types)
protein_locations = files.protein_location
protein_locations = np.array(protein_locations)
protein_locations = ([eval(p) for p in protein_locations])
train_protein_locations = np.array(protein_locations)[train_inds]

if config['classification']['protein_task'] == '5k':
    protein_to_num = protein_to_num_5k
elif config['classification']['protein_task'] == 'full':
    protein_to_num = protein_to_num_full

num_proteins = len(protein_to_num)
protein_target_matrix = np.zeros((len(all_features), num_proteins))
for i,c in enumerate(protein_locations):
    for class_ind in c:
        if class_ind in protein_to_num:
            protein_target_matrix[i, protein_to_num[class_ind]] = 1

cell_type_target_matrix = np.zeros((len(all_features), len(np.unique(cell_types))))
cell_type_to_num = {v : k for k,v in enumerate(np.unique(cell_types))}
for i,c in enumerate(cell_types):
    if c in cell_type_to_num.keys():
        cell_type_target_matrix[i, cell_type_to_num[c]] = 1
num_cells = len(cell_type_to_num)
print(f"working on {config['classification']['output_file']}")
print(f'classifying {num_proteins} protein localizations')

X_train = all_features[train_inds, :]
X_test = all_features[test_inds, :]
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

protein_train = protein_target_matrix[train_inds, :]
protein_test = protein_target_matrix[test_inds, :]
cell_type_train = cell_type_target_matrix[train_inds, :]
cell_type_test = cell_type_target_matrix[test_inds, :]

results = {}
def aggregate_result(x):
    result = np.zeros(x.shape[1])
    x = x.mean(axis=0)
    if x.max() == 0: return result
    result[x.argmax()] = 1
    return result

class Multilabel_classifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


def threshold_probabilities(prediction):
    return (prediction > 0.5).int()

from torchvision.ops import sigmoid_focal_loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def network_train(num_classes, X_train, y_train):
    if config['classification']['loss'] == 'focal':
        # criterion = FocalLoss()
        criterion = sigmoid_focal_loss
    else:
        criterion = nn.BCELoss()
    model = Multilabel_classifier(X_train.shape[1], y_train.shape[1])
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    batch_size = 1000
    num_epochs = 1000
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    batches = [(X_train[i : i + batch_size, :], y_train[i : i + batch_size, :])
            for i in range(0, X_train.shape[0], batch_size)]

    # pbar = tqdm(range(num_epochs))
    pbar = (range(num_epochs))
    running_loss = 0
    for epoch in pbar:
        temp_running_loss = 0
        for ind, (X_batch, y_batch) in enumerate(batches):
            optimizer.zero_grad()
            # loss = criterion(model(X_batch), y_batch, reduction='mean')
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            temp_running_loss += loss.item()
            optimizer.step()
        running_loss = temp_running_loss / len(batches)
        # if config['classification']['verbose']:
        #     pbar.set_description(f'epoch: {epoch}, running loss {running_loss:.2f}')
    return model.eval()

def network_predict(classifier, X_test):
    prediction = threshold_probabilities(classifier(torch.Tensor(X_test))).detach().numpy()
    return prediction

def network_save(classifier):
    torch.save(classifier.state_dict(), config['classification'][f'{task}_classifier'])

def network_load(classifier, X_train, y_train):
    model = Multilabel_classifier(X_train.shape[1], y_train.shape[1])
    model.load_state_dict(torch.load(config['classification'][f'{task}_classifier']))
    return model

def MLLR_train(num_classes, X_train, y_train):
    classifier = MLLR(max_iter=1000, num_classes = num_classes)
    classifier.fit(X_train, y_train)
    return classifier

def MLLR_predict(classifier, X_test):
    prediction = classifier.predict(X_test)
    return prediction

def MLLR_save(classifier):
    torch.save(classifier, config['classification'][f'{task}_classifier'])

def MLLR_load(num_classes, X_train, y_train):
    return torch.load(config['classification'][f'{task}_classifier'])

for task in ['protein', 'cell_type']:
    if config['classification'][f'train_{task}'] == False: continue
    if task == 'protein':
        num_classes = num_proteins
        y_test = protein_test
        y_train = protein_train
    elif task == 'cell_type':
        num_classes = num_cells
        y_test = cell_type_test
        y_train = cell_type_train

    inds_per_ID = {}
    prediction_per_ID_train = np.zeros((len(train_id_list), num_classes))
    target_matrix_per_ID_train = np.zeros((len(train_id_list), num_classes))
    for ID in tqdm(train_id_list):
        inds = np.where(files.iloc[train_inds].ID == ID)[0]
        if len(inds) == 0: continue
        inds_per_ID[ID] = inds

    prediction_per_ID_test = np.zeros((len(test_id_list), 5))
    target_matrix_per_ID_test = np.zeros((len(test_id_list), 5))
    for ID in tqdm(test_id_list):
        inds = np.where(files.iloc[test_inds].ID == ID)[0]
        if len(inds) == 0: continue
        inds_per_ID[ID] = inds

    def profile_features(X,y, inds_per_ID, IDs):
        new_X = []
        new_y = []
        for ID in IDs:
            new_X.append(X[inds_per_ID[ID]].mean(axis=0))
            new_y.append(y[inds_per_ID[ID]][0])
        return torch.Tensor(new_X), torch.Tensor(new_y)
    if config['classification']['predict_single_cell_profile']:
        profile_X_train, profile_y_train = profile_features(X_train, y_train, inds_per_ID, train_id_list)
        profile_X_test, profile_y_test = profile_features(X_test, y_test, inds_per_ID, test_id_list)
        if config['classification']['profiling_classifier_type'] == 'MLLR':
            if config['classification']['use_pretrained_protein_classifer']:
                profiling_classifier = MLLR_load(num_classes, profile_X_train, profile_y_train)
            else:
                profiling_classifier = MLLR_train(num_classes, profile_X_train, profile_y_train)
        elif config['classification']['profiling_classifier_type'] == 'network':
            if config['classification']['use_pretrained_protein_classifer']:
                profiling_classifier = network_load(num_classes, profile_X_train, profile_y_train)
            else:
                profiling_classifier = network_train(num_classes, profile_X_train, profile_y_train)

        if config['classification']['profiling_classifier_type'] == 'MLLR':
            test_prediction = MLLR_predict(profiling_classifier, profile_X_test)
            train_prediction = MLLR_predict(profiling_classifier, profile_X_train)
        elif config['classification']['profiling_classifier_type'] == 'network':
            test_prediction = network_predict(profiling_classifier, profile_X_test)
            train_prediction = network_predict(profiling_classifier, profile_X_train)

        test_f1_score = (f1_score(test_prediction, profile_y_test, average='macro'))
        train_f1_score = (f1_score(train_prediction, profile_y_train, average='macro'))
        results[f'{task}_profiling_classifier_whole_image_test'] = float(test_f1_score)
        results[f'{task}_profiling_classifier_whole_image_train'] = float(train_f1_score)
        print(f'train size {profile_X_train.shape}, test size {profile_X_test.shape}, {task} test {test_f1_score}, {task} train {train_f1_score}')
        torch.save(profiling_classifier, config['classification'][f'{task}_profiling_classifier'])


    if config['classification']['full_image_train']:
        if config['classification']['classifier_type'] == 'MLLR':
            if config['classification']['use_pretrained_protein_classifer']:
                classifier = MLLR_load(num_classes, X_train, y_train)
            else:
                classifier = MLLR_train(num_classes, X_train, y_train)
        elif config['classification']['classifier_type'] == 'network':
            if config['classification']['use_pretrained_protein_classifer']:
                classifier = network_load(num_classes, X_train, y_train)
            else:
                classifier = network_train(num_classes, X_train, y_train)

        if config['classification']['classifier_type'] == 'MLLR':
            test_prediction = MLLR_predict(classifier, X_test)
            train_prediction = MLLR_predict(classifier, X_train)
        elif config['classification']['classifier_type'] == 'network':
            test_prediction = network_predict(classifier, X_test)
            train_prediction = network_predict(classifier, X_train)

        test_f1_score = (f1_score(test_prediction, y_test, average='macro'))
        train_f1_score = (f1_score(train_prediction, y_train, average='macro'))
        results[f'{task}_classifier_whole_image_test'] = float(test_f1_score)
        results[f'{task}_classifier_whole_image_train'] = float(train_f1_score)
        print(f'train size {X_train.shape}, test size {X_test.shape}, {task} test {test_f1_score}, {task} train {train_f1_score}')
        torch.save(classifier, config['classification'][f'{task}_classifier'])


    if config['classification']['aggregate_result_by_single_cell']:
        prediction_per_ID_train = np.zeros((len(train_id_list), num_classes))
        target_matrix_per_ID_train = np.zeros((len(train_id_list), num_classes))
        ID_ind = 0
        for ID in tqdm(train_id_list):
            try:
                inds = inds_per_ID[ID]
            except:
                continue
            prediction_per_ID_train[ID_ind, :] = aggregate_result(train_prediction[inds])
            target_matrix_per_ID_train[ID_ind, :] = aggregate_result(y_train[inds])
            ID_ind += 1

        prediction_per_ID_test = np.zeros((len(test_id_list), 5))
        target_matrix_per_ID_test = np.zeros((len(test_id_list), 5))
        ID_ind = 0
        for ID in tqdm(test_id_list):
            try:
                inds = inds_per_ID[ID]
            except:
                continue
            prediction_per_ID_test[ID_ind, :] = aggregate_result(test_prediction[inds])
            target_matrix_per_ID_test[ID_ind, :] = aggregate_result(y_test[inds])
            ID_ind += 1

        train_f1_score = (f1_score(prediction_per_ID_train, target_matrix_per_ID_train, average='macro'))
        test_f1_score = (f1_score(prediction_per_ID_test, target_matrix_per_ID_test, average='macro'))
        print(f'train size {prediction_per_ID_train.shape}, test size {prediction_per_ID_test.shape}, {task} test {test_f1_score}, {task} train {train_f1_score}')
        results[f'{task}_classifier_single_cell_test'] = float(test_f1_score)
        results[f'{task}_classifier_single_cell_train'] = float(train_f1_score)

with open(config['classification']['output_file'], 'w') as outfile:
    yaml.dump(results, outfile, default_flow_style=False)

