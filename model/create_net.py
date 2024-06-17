import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from model.Resnet import resnet50
from model.FusionModel import FusionModel
import os


def adjust_learning_rate_new(optimizer, decay=0.5):  # 调整优化器的学习率
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']

def create_net(config_dict):
    if config_dict["first_device"] >= 0:
        if config_dict["method"] == 'ResNet':
            cnn = resnet50().cuda()  # 创建一个 ResNet-50 模型的实例 cnn，并将其移动到 GPU 上进行加速计算
        else:
            cnn = FusionModel(num_classes=config_dict["num_classes"], num_count=65).cuda()
        cudnn.benchmark = True  # cuDNN 库进行自动优化
    else:
        if config_dict["method"] == 'ResNet':
            cnn = resnet50()
        else:
            cnn = FusionModel(num_classes=config_dict["num_classes"], num_count=65)
    
    params = []
    new_param_names = ['fc', 'counting']  # 用于选出resnet50中最后的全连接层和counting组件
    for key, value in dict(cnn.named_parameters()).items():  # 遍历模型 cnn 的所有命名参数
        if value.requires_grad:  # 检查当前参数是否需要进行梯度计算和优化
            if any(i in key for i in new_param_names):  # 检查当前参数的名称是否包含在 new_param_names 列表中
                params += [{'params': [value], 'lr': config_dict["lr"] * 0.5, 'weight_decay': config_dict["weight_decay"]}]  # 设置较低的学习率和较高的权重衰减
            else:
                params += [{'params': [value], 'lr': config_dict["lr"] * 1.0, 'weight_decay': config_dict["weight_decay"]}]  # 原代码此处有误
          
    if config_dict["method"] == 'ResNet':
        config_dict["optimizer"] = torch.optim.SGD(params, momentum=0.9)  # SGD优化
    else:
        config_dict["optimizer"] = torch.optim.Adam(cnn.parameters(), lr=config_dict["lr"])
    
    if os.path.exists(config_dict["model_path"]):
        cnn.load_state_dict(config_dict["model_path"])
        
    config_dict["loss_func"] = nn.CrossEntropyLoss().cuda()  # 交叉熵损失函数
    config_dict["kl_loss_1"] = nn.KLDivLoss().cuda()  # 对于三个标签
    config_dict["kl_loss_2"] = nn.KLDivLoss().cuda()  # 测量概率分布之间的差异
    config_dict["kl_loss_3"] = nn.KLDivLoss().cuda()  # 处理多个目标分布而创建的多个实例
    
    config_dict["cnn"] = cnn

