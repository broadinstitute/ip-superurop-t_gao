import torch
from torch import Tensor
from torch import nn
import logging as log
from typing import Optional # required for "Optional[type]"

class TorchStandardScaler(object):
    """
    Standard scalar transform.
    Inspired by: https://discuss.pytorch.org/t/pytorch-tensor-scaling/38576/8 
    """
    def __call__(self, x):
        Xmean = x.mean((-2,-1), keepdim=True)
        Xstd = x.std((-2,-1), unbiased=False, keepdim=True)
        x -= Xmean
        x /= (Xstd + 1e-7)
        return x

## The code below gives you Flatten and the double Adaptive Pooling (from fastai), plus
## a viable head. Mind that you got to fill the number of FC's nodes manually
# Source: https://forums.fast.ai/t/solved-using-a-fastai-trained-model-with-plain-pytorch/42861/24

class Flatten(nn.Module):
    "Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"
    def __init__(self, full:bool=False):
        super().__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)

class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`." # from pytorch
    def __init__(self, sz:Optional[int]=None): 
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
    
def FAHead(num_feats, num_classes):
    return \
        nn.Sequential(        
            AdaptiveConcatPool2d(),
            Flatten(),
            nn.BatchNorm1d(num_feats),
            nn.Dropout(p=0.25),
            nn.Linear(num_feats, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.25),
            nn.Linear(512, num_classes),
        )


# For making balanced dataloaders
def make_weights_for_balanced_classes(images, nclasses):                        
    count = images.value_counts()                                                   
    N = float(count.sum())
    weight = [0]*len(images)
    for idx, val in enumerate(images):                                          
        weight[idx] = N/count[val]                                  
    return weight

# A to add a hook for collecting outputs
# in arbitrary layers
def myHook(module, inputs, outputs):
    return outputs, inputs[0]
#Source: https://github.com/janfreyberg/pytorch-revgrad/tree/alpha_parameter
from torch.autograd import Function
from torch.nn import Module

class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None



class RevGradM(Module):
    def __init__(self, alpha=0.5, *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return revgrad(input_, self._alpha)
    