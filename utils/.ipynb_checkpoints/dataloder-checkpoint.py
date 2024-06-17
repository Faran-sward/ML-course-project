from torchvision import transforms
from torch.utils.data import DataLoader
from utils.dataset_processing import DatasetProcessing

def load_data(config_dict, i):
    dset_train = dataset_processing.DatasetProcessing(config_dict["data_path"], config_dict["train_files"][i], config_dict["num_classes"], 
                                                      transform=transforms.Compose([
            transforms.Resize((config_dict["image_resize"], config_dict["image_resize"])),  # 旧版本的 torchvision 中的方法为Scale
            transforms.RandomCrop(config_dict["train_size"]),  # 随机从图像中裁剪出大小为 (224, 224) 的区域
            transforms.RandomHorizontalFlip(),  # 以一定的概率对图像进行水平翻转
            transforms.ToTensor(),  # 图像数据转换为 PyTorch 中的张量（Tensor）格式。它还会将像素值归一化到 [0, 1] 的范围内
            RandomRotate(rotation_range=config_dict["image_rotation"]),  # 随机旋转图像，角度的范围为 -20 到 +20 度之间
            config_dict["normalize"],  # 给定的均值和标准差进行归一化
        ]))

    dset_test = dataset_processing.DatasetProcessing(["data_path"], ["test_files"][i], config_dict["num_classes"], 
                                                     transform=transforms.Compose([
            transforms.Resize((["train_size"], ["train_size"])),
            transforms.ToTensor(),
            config_dict["normalize"],
        ]))

    config_dict["train_loader"] = DataLoader(dset_train,
                              batch_size=config_dict["batch_size"],
                              shuffle=True,  # 是否在每个epoch打乱样本的顺序
                              num_workers=config_dict["num_workers"],
                              pin_memory=True)  # 是否将数据加载到固定内存

    config_dict["test_loader"] = DataLoader(dset_test,
                             batch_size=config_dict["batch_size_test"],
                             shuffle=False,
                             num_workers=config_dict["num_workers"],
                             pin_memory=True)

