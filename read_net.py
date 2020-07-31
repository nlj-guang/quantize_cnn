import argparse
import torchvision.models as models
import warnings
import random
import os
import math
import numpy as np
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils.train_val import train, save_checkpoint, validate
from utils.data_loader import load_val_data_299, load_val_data
from utils.get_cifar_data import cifar_train_data, cifar_val_data
from quantize.quantize_module_ import QWConv2D, QWAConv2D, QWALinear, Scalar
from tensorboardX import SummaryWriter
from feactures.read_feature2 import get_feature1,get_feature2
from utils.load_model import load_vgg2, load_resnet

def main():
    torch.set_printoptions(precision = 16)
    epsilon = 1e-5
    data_set = "cifar10"
    criterion = torch.nn.CrossEntropyLoss().cuda()
    val_loader = cifar_val_data(data_set, 50, 8)

    model, model2 = load_resnet(data_set,"resnet34")

    model.eval()
    validate(model, val_loader, criterion)
    model2.eval()
    validate(model2, val_loader, criterion)


    return

if __name__ == "__main__":
    main()