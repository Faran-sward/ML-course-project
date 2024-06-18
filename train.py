from utils.utils import  AverageMeter
from utils.genLD import generate_distribution
from utils.report import calculate_metrics, calculate_mae_mse
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
        for step, (image, _, counting) in enumerate(config_dict["train_loader"]):   # gives batch data, normalize x when iterate train_loader
            # image 表示当前批次的输入数据
            # counting 表示当前批次的计数信息
            image = image.cuda()
            counting = counting.numpy()

            # generating counting_distribution
            counting = counting - 1  # ？
            counting_distribution = generate_distribution(counting, config_dict["sigma"], 'klloss', 65)
            # 计算各个类别的总和
            grading_distribution = np.vstack((np.sum(counting_distribution[:, :5], 1), np.sum(counting_distribution[:, 5:20], 1), np.sum(counting_distribution[:, 20:50], 1), np.sum(counting_distribution[:, 50:], 1))).transpose()
            counting_distribution = torch.from_numpy(counting_distribution).cuda().float()
            grading_distribution = torch.from_numpy(grading_distribution).cuda().float()

            # train
            config_dict["cnn"].train()

            cls, cou, cou2cls = config_dict["cnn"](image, None)  # nn output
            loss_cls = config_dict["kl_loss_1"](torch.log(cls), grading_distribution) * 4.0
            loss_cou = config_dict["kl_loss_2"](torch.log(cou), counting_distribution) * 65.0
            loss_cls_cou = config_dict["kl_loss_3"](torch.log(cou2cls), grading_distribution) * 4.0
            loss = (loss_cls + loss_cls_cou) * 0.5 * config_dict["lam"] + loss_cou * (1.0 - config_dict["lam"])
            
            config_dict["writer"].add_scalar('loss_cls_' + str(config_dict["cross_validation_idx"]), loss_cls, epoch * len(config_dict["train_loader"]) + step)
            config_dict["writer"].add_scalar('loss_cou_' + str(config_dict["cross_validation_idx"]), loss_cou, epoch * len(config_dict["train_loader"]) + step)
            config_dict["writer"].add_scalar('loss_cls_cou_' + str(config_dict["cross_validation_idx"]), loss_cls_cou, epoch * len(config_dict["train_loader"]) + step)
            config_dict["writer"].add_scalar('loss_' + str(config_dict["cross_validation_idx"]), loss, epoch * len(config_dict["train_loader"]) + step)
            
            config_dict["optimizer"].zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            config_dict["optimizer"].step()                # apply gradients

            losses_cls.update(loss_cls.item(), image.size(0))
            losses_cou.update(loss_cou.item(), image.size(0))
            losses_cou2cls.update(loss_cls_cou.item(), image.size(0))
            losses.update(loss.item(), image.size(0))  # 更新损失函数的累计值
            
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
        grading_gt = np.array([])
        counting_pred = np.array([])
        merge_gt = np.array([])
        counting_gt = np.array([])
        grading_pred = np.array([])
        config_dict["cnn"].eval()  # 将模型设置为评估模式，来确保模型在测试阶段
        
        for step, (image, grading, counting) in enumerate(config_dict["test_loader"]):   # gives batch data, normalize x when iterate train_loader

            image = image.cuda()
            grading = grading.cuda()

            grading_gt = np.hstack((grading_gt, grading.data.cpu().numpy()))
            counting_gt = np.hstack((counting_gt, counting.data.cpu().numpy()))

            config_dict["cnn"].eval()

            cls, cou, cou2cls = config_dict["cnn"](image, None)

            loss = config_dict["loss_func"](cou2cls, grading)
            test_loss += loss.data

            _, merge_pred = torch.max(cls + cou2cls, 1)
            _, preds = torch.max(cls, 1)
            # preds = preds.data.cpu().numpy()
            counting_pred = np.hstack((counting_pred, preds.data.cpu().numpy()))
            merge_gt = np.hstack((merge_gt, merge_pred.data.cpu().numpy()))

            _, preds_l = torch.max(cou, 1)
            preds_l = (preds_l + 1).data.cpu().numpy()
            # preds_l = cou2cou.data.cpu().numpy()
            grading_pred = np.hstack((grading_pred, preds_l))

            batch_corrects = torch.sum((preds == grading)).data.cpu().numpy()
            test_corrects += batch_corrects
            
            config_dict["writer"].add_scalar('test_loss_' + str(config_dict["cross_validation_idx"]), loss, config_dict["evaluate_counter"])
            config_dict["evaluate_counter"] = config_dict["evaluate_counter"] + 1


        test_loss = test_loss.float() / len(config_dict["test_loader"])
        test_acc = test_corrects / len(config_dict["test_loader"].dataset)#3292  #len(test_loader)
        

        message = '%s %6.1f | %0.3f | %0.3f\n' % ( \
                "test ", config_dict["epoch"],
                test_loss.data,
                test_acc)

        _, _, pre_se_sp_yi_report = calculate_metrics(counting_pred, grading_gt)
        _, _, pre_se_sp_yi_report_m = calculate_metrics(merge_gt, grading_gt)
        _, MAE, MSE, mae_mse_report = calculate_mae_mse(counting_gt, grading_pred, grading_gt)

        if True:
            config_dict["_log"].write(str(pre_se_sp_yi_report) + '\n')
            config_dict["m_log"].write(str(pre_se_sp_yi_report_m) + '\n')
            config_dict["e_log"].write(str(mae_mse_report) + '\n')
            config_dict["log"].write(message + '\n')