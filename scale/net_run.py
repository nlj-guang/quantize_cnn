# coding=utf-8
"""
    resnet_net
"""
import torch
from torchvision import models
import numpy as np
import os
from utils.get_cifar_data import  cifar_val_data
from utils.data_loader import load_val_data
from utils.train_val import validate
from utils.load_model import load_resnet, load_vgg, load_vgg2
from vgg import vgg_scale2,vgg

torch.set_printoptions(precision = 8)
gpu = [0]
epsilon = 1e-5

def main():
    val_loader = load_val_data(data_dir="", batch_size=200)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    model = models.densenet121(pretrained=True)

    model = torch.nn.DataParallel(model)
    model.cuda()
    print(model)
    model.eval()
    validate(model, val_loader, criterion)


if __name__ == '__main__':
    main()