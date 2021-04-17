import torch
import numpy as np
import torch.nn as nn
from torch.nn import init


def weights_init_normal(model):
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            init.normal_(m.weight.data,0.0, 0.02)
        elif isinstance(m,nn.ConvTranspose2d):
            init.normal_(m.weight.data,0.0,0.02)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data,1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight.data)


def weights_init_xavier(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # print('dim',m.weight.data.dim())
            init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m,nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)


def weights_init_kaiming(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data)
        elif isinstance(m,nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight.data)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data)
            init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight.data)


def init_weights(net, init_type='kaiming'):
    if init_type == 'normal':
        weights_init_normal(net)
    elif init_type == 'xavier':
        weights_init_xavier(net)
    elif init_type == 'kaiming':
        # 何凯明初始化方法比较适用于ReLU激活函数，之前用的是xavier初始化方法，刚换成的Hekaiming初始化方法
        weights_init_kaiming(net)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized  =False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self,val,weight):
        self.val = val
        self.avg = val
        self.sum = val*weight
        self.count = True
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def make_one_hot(input, num_classes=None, smoothing=0.0):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Shapes:
        predict: A tensor of shape [N, *] without sigmoid activation function applied
        target: A tensor of shape same with predict
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    assert 0<=smoothing<1
    confidence = 1.0-smoothing

    if num_classes is None:
        num_classes = input.max() + 1

    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.empty(shape)
    # result.fill_(smoothing/(num_classes-1))
    result = result.scatter_(1, input, 1)
    return result