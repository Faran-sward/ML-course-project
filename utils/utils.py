import sys
from torch.nn import init
import torch.nn as nn


# evaluate meters
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# print logger
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)

    else:
        raise NotImplementedError
        # 提醒开发者某个方法或功能需要进一步实现，并作为一个标记，以便后续的代码编写和调试


def weights_init(m):  # 初始化神经网络模型中的权重（weights）和偏置（biases）
    if isinstance(m, nn.Conv2d):  # 如果是卷积层
        init.kaiming_normal(m.weight, mode='fan_out')  # 使用 Kaiming 正态分布初始化
        if m.bias is not None:  # 如果卷积层有偏置项
            init.constant(m.bias, 0)  # 将偏置项设置为 0
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):  # 如果是归一化层
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)
    elif isinstance(m, nn.Linear):  # 如果是全连接层
        init.normal(m.weight, std=0.001)  # 使用普通的正态分布初始化
        if m.bias is not None:
            init.constant(m.bias, 0)
