import torch
from torch import nn
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch
import pretrainedmodels
from pytorchcv.model_provider import get_model as ptcv_get_model


class Xception_osmr(nn.Module):
    def __init__(self, classCount):
        super(Xception_osmr, self).__init__()
        self.model_ft = ptcv_get_model("xception", pretrained=True)
        num_ftrs = self.model_ft.output.in_features
        self.model_ft.features.final_block.pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.output = nn.Identity()
    def forward(self, x):
        x = self.model_ft(x)
        return x

class se_resnext50_32x4d(nn.Module):
    def __init__(self, classCount):
        super(se_resnext50_32x4d, self).__init__()
        self.model_ft = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet')
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.last_linear = nn.Identity()
    def forward(self, x):
        x = self.model_ft(x)
        return x

class DenseNet169_change_avg(nn.Module):
    def __init__(self, classCount, isTrained=False):
        super(DenseNet169_change_avg, self).__init__()
        self.densenet169 = torchvision.models.densenet169(pretrained=isTrained).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.densenet169(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class DenseNet121_change_avg(nn.Module):
    def __init__(self, classCount, isTrained=False):
        super(DenseNet121_change_avg, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.densenet121(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)

        return x

class ibn_densenet121_osmr(nn.Module):
    def __init__(self, classCount):
        super(ibn_densenet121_osmr, self).__init__()
        self.model_ft = ptcv_get_model("ibn_densenet121", pretrained=True)
        num_ftrs = self.model_ft.output.in_features
        self.model_ft.features.final_pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.output = nn.Identity()
    def forward(self, x):
        x = self.model_ft(x)
        return x


def get_model(backbone):
    if backbone == 'DenseNet121_change_avg':
        model = DenseNet121_change_avg(28, True)
    elif backbone == 'ibn_densenet121_osmr':
        model = ibn_densenet121_osmr(28)
    elif backbone == 'DenseNet169_change_avg':
        model = DenseNet169_change_avg(28, True)
    elif backbone == 'se_resnext50_32x4d':
        model = se_resnext50_32x4d(28)
    elif backbone == 'Xception_osmr':
        model = Xception_osmr(28)
    return model

model_name_list = ['DenseNet121_change_avg_512_all_more_train_add_3_v2',
                   'DenseNet121_change_avg_512_all_more_train_add_3_v3',
                   'DenseNet121_change_avg_512_all_more_train_add_3_v5',
                   'DenseNet169_change_avg_512_all_more_train_add_3_v5',
                   'se_resnext50_32x4d_512_all_more_train_add_3_v5',
                   'Xception_osmr_512_all_more_train_add_3_v5',
                   'ibn_densenet121_osmr_512_all_more_train_add_3_v5_2']

def get_wair_model(model_name, fold = 0):
    trained_model_snapshot_path = '/dgx1nas1/storage/data/mdoron/human_protein_atlas/kaggle_model/wair/models/' + model_name + '/'
    mm_name = ''
    if 'DenseNet121_change_avg' in model_name:
        mm_name = 'DenseNet121_change_avg'
    elif 'DenseNet169_change_avg' in model_name:
        mm_name = 'DenseNet169_change_avg'
    elif 'se_resnext50_32x4d' in model_name:
        mm_name = 'se_resnext50_32x4d'
    elif 'Xception_osmr' in model_name:
        mm_name = 'Xception_osmr'
    elif 'ibn_densenet121_osmr' in model_name:
        mm_name = 'ibn_densenet121_osmr'
    model = get_model(mm_name)
    model = nn.DataParallel(model).cuda()
    state = torch.load(trained_model_snapshot_path + 'model_min_loss_{fold}.pth.tar'.format(fold=fold))
    model.load_state_dict(state['state_dict'], strict=False)
    return model.module
