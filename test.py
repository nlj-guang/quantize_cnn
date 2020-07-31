from utils.train_val import train, save_checkpoint, validate
from utils.data_loader import load_val_data_299, load_val_data, load_train_data
from utils.get_cifar_data import cifar_train_data, cifar_val_data
from torchvision import datasets
import os
import numpy as np
import random
import torch
import torchvision.models as models
from utils.load_model import load_vgg2, load_resnet
from feactures.read_feature2 import get_feature1, get_feature2

def main():
    torch.set_printoptions(precision = 16)
    data_set = "cifar10"
    criterion = torch.nn.CrossEntropyLoss().cuda()
    train_loader, train_sampler= cifar_train_data(data_set, 200, 8, distributed=False)
    val_loader = cifar_val_data(data_set, 200, 8)

    model, model2 = load_resnet(data_set,"resnet18")

    cnt = random.randint(1, 600)
    print(cnt)
    i = 0
    for t_image,lable in train_loader:
        if i>=cnt:
            break
        i = i + 1
    print(t_image.size())
    cnt = random.randint(1, 50)
    i = 0
    for v_image, lable in val_loader:
        if i >= cnt:
            break
        i = i + 1

    modules = model.named_modules()
    s = []
    for name, module in modules:
        if name[-5:-1] == 'conv':
            s.append(name)

    s1 = ['features.module.0', 'features.module.3', 'features.module.7', 'features.module.10','features.module.14',
         'features.module.17', 'features.module.20','features.module.24','features.module.27','features.module.30',
         'features.module.34', 'features.module.37','classifier.0']

    s2 = ['features.module.0', 'features.module.2', 'features.module.5', 'features.module.7', 'features.module.10',
         'features.module.12', 'features.module.14','features.module.17','features.module.19','features.module.21',
         'features.module.24', 'features.module.26','features.module.28','classifier.0']

    for key in s:
        l = get_feature2(model, key, t_image)
        l2 = get_feature2(model2, key, t_image)
        print(key)
        print(l.size())
        print(torch.max(l))
        print(torch.max(l2))
        print(torch.min(l))
        print(torch.min(l2))
        i = i+1

    return

if __name__ == "__main__":
    main()


