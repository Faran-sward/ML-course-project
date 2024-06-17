from utils.utils import  AverageMeter
from utils.genLD import genLD
from utils.report import report_precision_se_sp_yi, report_mae_mse
from timeit import default_timer as timer
from model.create_net import adjust_learning_rate_new
import torch
import numpy as np
from utils.utils import time_to_str
from timeit import default_timer as timer
import os


def train(config_dict):
    # training and testing
    config_dict["start"] = timer()
    
    for epoch in range(config_dict["lr_steps"][-1]):#(EPOCH):#
        config_dict["epoch"] = epoch
        
        if epoch in config_dict["lr_steps"]:  # 每 30 个epoch调整学习率
            adjust_learning_rate_new(config_dict["optimizer"], 0.5)
        # scheduler.step(epoch)

        losses_cls = AverageMeter()
        losses_cou = AverageMeter()
        losses_cou2cls = AverageMeter()
        losses = AverageMeter()  # 记录计数结果，分类结果，预测结果的损失
        # '''
        config_dict["cnn"].train()  # 将模型设置为训练模式，然后迭代训练数据
        for step, (b_x, b_y, b_l) in enumerate(config_dict["train_loader"]):   # gives batch data, normalize x when iterate train_loader
            # b_x 表示当前批次的输入数据
            # b_y 表示当前批次的标签数据
            # b_l 表示当前批次的计数信息
            b_x = b_x.cuda()
            b_l = b_l.numpy()

            # generating ld
            b_l = b_l - 1  # ？
            ld = genLD(b_l, config_dict["sigma"], 'klloss', 65)
            # 计算各个类别的总和
            ld_4 = np.vstack((np.sum(ld[:, :5], 1), np.sum(ld[:, 5:20], 1), np.sum(ld[:, 20:50], 1), np.sum(ld[:, 50:], 1))).transpose()
            ld = torch.from_numpy(ld).cuda().float()
            ld_4 = torch.from_numpy(ld_4).cuda().float()

            # train
            config_dict["cnn"].train()

            cls, cou, cou2cls = config_dict["cnn"](b_x, None)  # nn output
            loss_cls = config_dict["kl_loss_1"](torch.log(cls), ld_4) * 4.0
            loss_cou = config_dict["kl_loss_2"](torch.log(cou), ld) * 65.0
            loss_cls_cou = config_dict["kl_loss_3"](torch.log(cou2cls), ld_4) * 4.0
            loss = (loss_cls + loss_cls_cou) * 0.5 * config_dict["lam"] + loss_cou * (1.0 - config_dict["lam"])
            
            config_dict["writer"].add_scalar('loss_cls_' + str(config_dict["cross_validation_idx"]), loss_cls, epoch * len(config_dict["train_loader"]) + step)
            config_dict["writer"].add_scalar('loss_cou_' + str(config_dict["cross_validation_idx"]), loss_cou, epoch * len(config_dict["train_loader"]) + step)
            config_dict["writer"].add_scalar('loss_cls_cou_' + str(config_dict["cross_validation_idx"]), loss_cls_cou, epoch * len(config_dict["train_loader"]) + step)
            config_dict["writer"].add_scalar('loss_' + str(config_dict["cross_validation_idx"]), loss, epoch * len(config_dict["train_loader"]) + step)
            
            config_dict["optimizer"].zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            config_dict["optimizer"].step()                # apply gradients

            losses_cls.update(loss_cls.item(), b_x.size(0))
            losses_cou.update(loss_cou.item(), b_x.size(0))
            losses_cou2cls.update(loss_cls_cou.item(), b_x.size(0))
            losses.update(loss.item(), b_x.size(0))  # 更新损失函数的累计值
            
            message = '%s %6.0f |%s %4.0f | %0.3f | %0.3f | %0.3f | %0.3f | %s\n' % ( \
                "train", config_dict["epoch"],
                "step", step,
                losses_cls.avg,
                losses_cou.avg,
                losses_cou2cls.avg,
                losses.avg,
                time_to_str((timer() - config_dict["start"]), 'min'))
            # print(message)
            config_dict['log'].write(message)
            
        if (epoch + 1) % config_dict["frequency"] == 0:
            evaluate(config_dict)
            
        model_name = 'model_' + str(config_dict["cross_validation_idx"]) + '.pth'
        torch.save(model_name, os.path.join(config_dict["model_dir"], model_name))

            
            

def evaluate(config_dict):
    with torch.no_grad():
        test_loss = 0
        test_corrects = 0  # 根据模型的预测结果和真实标签计算准确率
        y_true = np.array([])
        y_pred = np.array([])
        y_pred_m = np.array([])
        l_true = np.array([])
        l_pred = np.array([])
        config_dict["cnn"].eval()  # 将模型设置为评估模式，来确保模型在测试阶段
        
        for step, (test_x, test_y, test_l) in enumerate(config_dict["test_loader"]):   # gives batch data, normalize x when iterate train_loader

            test_x = test_x.cuda()
            test_y = test_y.cuda()

            y_true = np.hstack((y_true, test_y.data.cpu().numpy()))
            l_true = np.hstack((l_true, test_l.data.cpu().numpy()))

            config_dict["cnn"].eval()

            cls, cou, cou2cls = config_dict["cnn"](test_x, None)

            loss = config_dict["loss_func"](cou2cls, test_y)
            test_loss += loss.data

            _, preds_m = torch.max(cls + cou2cls, 1)
            _, preds = torch.max(cls, 1)
            # preds = preds.data.cpu().numpy()
            y_pred = np.hstack((y_pred, preds.data.cpu().numpy()))
            y_pred_m = np.hstack((y_pred_m, preds_m.data.cpu().numpy()))

            _, preds_l = torch.max(cou, 1)
            preds_l = (preds_l + 1).data.cpu().numpy()
            # preds_l = cou2cou.data.cpu().numpy()
            l_pred = np.hstack((l_pred, preds_l))

            batch_corrects = torch.sum((preds == test_y)).data.cpu().numpy()
            test_corrects += batch_corrects
            
            config_dict["writer"].add_scalar('test_loss_' + str(config_dict["cross_validation_idx"]), loss, config_dict["evaluate_counter"])
            config_dict["evaluate_counter"] = config_dict["evaluate_counter"] + 1


        test_loss = test_loss.float() / len(config_dict["test_loader"])
        test_acc = test_corrects / len(config_dict["test_loader"].dataset)#3292  #len(test_loader)
        

        message = '%s %6.1f | %0.3f | %0.3f\n' % ( \
                "test ", config_dict["epoch"],
                test_loss.data,
                test_acc)

        _, _, pre_se_sp_yi_report = report_precision_se_sp_yi(y_pred, y_true)
        _, _, pre_se_sp_yi_report_m = report_precision_se_sp_yi(y_pred_m, y_true)
        _, MAE, MSE, mae_mse_report = report_mae_mse(l_true, l_pred, y_true)

        if True:
            config_dict["_log"].write(str(pre_se_sp_yi_report) + '\n')
            config_dict["m_log"].write(str(pre_se_sp_yi_report_m) + '\n')
            config_dict["e_log"].write(str(mae_mse_report) + '\n')
            config_dict["log"].write(message + '\n')