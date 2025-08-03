import torch
import numpy as np
import os
import random
from tqdm import tqdm
from torchvision import datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
import pandas as pd

import config


def set_seed(seed):
    if seed != 0:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def build_dataset(dataset_name, batch_size=256, num_workers=8):
    if dataset_name == "cifar10":
        testset = datasets.CIFAR10(root=config.DATA_CIFAR10_ROOT, train=False, transform=config.TRANSFORM_CIFAR10_TEST)
        # testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, num_workers=num_workers,
                                                 # shuffle=True)
        cal_num = config.CIFAR10_CALNUM
        test_num = config.CIFAR10_TESTNUM
        calibset, testset = torch.utils.data.random_split(testset, [cal_num, test_num])
        calibloader = torch.utils.data.DataLoader(dataset=calibset, batch_size=batch_size, num_workers=num_workers,
                                                  shuffle=True)
        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, num_workers=num_workers,
                                                 shuffle=True)

    return calibloader, testloader


def build_model(model_name, dataset_name, device):
    if dataset_name == "cifar10":
        from model.cifar10.resnet import ResNet18
        model = ResNet18()
        checkpoint = torch.load(config.RESNET18_CIFAR10_ROOT)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})

    model.eval()
    return model.to(device)
