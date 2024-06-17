import argparse
import os.path
from torchvision import transforms
from utils.args_util import get_config
from utils.dataloder import load_data
from model.create_net import create_net
from utils.utils import Logger
from train import train, evaluate
from torch.utils.tensorboard import SummaryWriter

def print_config(vdict, name="config"):
    """
    :param vdict: dict, 待打印的字典
    :param name: str, 打印的字典名称
    :return: None
    """
    print("-----------------------------------------")
    print("|This is the summary of {}:".format(name))
    var = vdict
    for i in var:
        if var[i] is None:
            continue
        print("|{:11}\t: {}".format(i, var[i]))
    print("-----------------------------------------")


def print_args(args, name="args"):
    """
    :param args:
    :param name: str, 打印的字典名称
    :return: None
    """
    print("-----------------------------------------")
    print("|This is the summary of {}:".format(name))
    for arg in vars(args):
        print("| {:<11} : {}".format(arg, getattr(args, arg)))
    print("-----------------------------------------")


def add_args_to_config(config, args):
    for arg in vars(args):
        # print("| {:<11} : {}".format(arg, getattr(args, arg)))
        config[str(arg)] = getattr(args, arg)


def main(args):
    # 获取方法所用的参数
    config = get_config(args.config_path)
    add_args_to_config(config, args)
    print_config(config)
    
    config["data_path"] = os.path.join(config["dataset_path"], 'JPEGImages')
    config["train_files"] = [os.path.join(config["dataset_path"], 'NNEW_trainval_' + str(cross_val_index) + '.txt') for cross_val_index in range(config["cross_validation"])]
    config["test_files"] = [os.path.join(config["dataset_path"], 'NNEW_test_' + str(cross_val_index) + '.txt') for cross_val_index in range(config["cross_validation"])]
    # 计算数据集中各通道的平均值和标准差，然后将这些数值用于归一化
    config["normalize"] = transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266],
                                     std=[0.2814769, 0.226306, 0.20132513])
    config["lr_steps"] = [i for i in range(0, config["num_epochs"] + 1, config["lr_steps"]) if i > 0]
    
    config["log"] = Logger()
    config["log"].open(os.path.join(config["save_dir"], 'log.txt'), mode="a")
    
    config["_log"] = Logger()
    config["_log"].open(os.path.join(config["save_dir"], '_log.txt'), mode="a")
    
    config["m_log"] = Logger()
    config["m_log"].open(os.path.join(config["save_dir"], 'm_log.txt'), mode="a")
    
    config["e_log"] = Logger()
    config["e_log"].open(os.path.join(config["save_dir"], 'e_log.txt'), mode="a")
    
    config["writer"] = SummaryWriter(log_dir=config["save_dir"])  # 可以指定log存储的目录

    
    for i in range(config["cross_validation"]):
        config["cross_validation_idx"] = i  
        config["evaluate_counter"] = 1
        
        load_data(config, i)
        
        create_net(config)
        
        train(config)
        
        # evaluate(config_dict=config)
        

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="")

    # 添加超参数
    parser.add_argument("--method", default="FusionModel", choices=["FusionModel", "ResNet"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batch_size_test", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--config_path", type=str, default="./configs/config.yaml")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="./data")
    parser.add_argument("--cross_validation", type=int, default=5)
    parser.add_argument("--image_rotation", type=int, default=20)
    parser.add_argument("--image_resize", type=int, default=256)
    parser.add_argument("--train_size", type=int, default=224)
    parser.add_argument("--num_epochs", type=int, default=120)
    parser.add_argument("--lr_steps", type=int, default=30)
    parser.add_argument("--sigma", type=float, default=30 * 0.1)
    parser.add_argument("--lam", type=float, default=6 * 0.1)
    parser.add_argument("--frequency", type=int, default=30)
    

    # 解析命令行参数
    main_args = parser.parse_args()

    # 调用主函数
    main(main_args)