import argparse
import torchvision.models as models
import warnings
import random
import os
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils.train_val import train, save_checkpoint, validate
from utils.get_cifar_data import mnist_train_data, mnist_val_data
from tensorboardX import SummaryWriter
from mnist.Le_5_Net import Lenet

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--workers', default=8, type=int, metavar='N',  # 修改为电脑cpu支持的线程数
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
# 如果是验证模型, 设置为True就好, 训练时值为False
parser.add_argument('--evaluate', default=False, type=bool,
                    help='evaluate model on validation set')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--device-ids', default=[0], type=int, nargs='+',
                    help='GPU ids to be used e.g 0 1 2 3')

parser.add_argument('--save-dir', default='model', type=str, help='directory to save trained model', required=True)
# l1 norm balance 设置为1或者0.1比较好, l2 norm balance 设置为100(~0.034) ~ 500 比较好
parser.add_argument('--norm', default=1, type=int, help='feature map norm, default 1')
parser.add_argument('--balance', default=100, type=float, help='balancing parameter (default: 100)')
# 论文中初始学习率 0.001, 每 10 epoch 除以 10, 这在只量化权重时候可以
# 在同时量化权重和激活时, 当使用0.001时, 我们可以观测到权重的持续上升
# 或许可以将初始学习率调为 0.01, 甚至 0.1
# guidance 方法中, 全精度模型的的学习率要小一些, 模型已经训练的很好了, 微调而已
# 不过来低精度模型的学习率可以调高一点
parser.add_argument('--lr', default=0.001, type=float,  # 论文中初始学习率 0.001, 每 10 epoch 除以 10
                    help='initial learning rate')

parser.add_argument('--lr-step', default=6, type=int, help='learning rate step scheduler')

args = parser.parse_args()
best_prec1 = 0
data_set = 'cifar10'

def main():
    global best_prec1

    print("\n"
          "=> init_lr      {: <20}\n"
          "=> lr-step      {: <20}\n"
          "=> momentum     {: <20}\n"
          "=> weight-decay {: <20}\n"
          "=> batch-size   {: <20}\n"
          "=> balance      {: <20}\n"
          "=> save-dir     {: <20}\n".format(
        args.lr, args.lr_step, args.momentum, args.weight_decay,
        args.batch_size, args.balance, args.save_dir))

    model = Lenet()
    model = torch.nn.DataParallel(model)
    model.cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step)
#    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [9, 19])


    val_loader = mnist_val_data(data_set, batch_size=100, workers=args.workers)
    args.distributed = False
    train_loader = mnist_train_data(data_set, args.batch_size, args.workers)

    if args.evaluate:
        path = os.path.join(args.save_dir, 'model_best.pth.tar')
        if os.path.isfile(path):
            init_model = torch.load(path)
            print(init_model['best_prec1'])
            print("Loaded evaluate model")
            model.load_state_dict(init_model['state_dict'])
        else:
            print("No evaluate mode found")
            return
        validate(model, val_loader, criterion, args.gpu)
        return

    summary_writer = SummaryWriter(args.save_dir)
    for epoch in range(args.start_epoch, args.epochs):
        lr_scheduler.step()

        # train for one epoch
        train(model, train_loader, criterion, optimizer, args.gpu, epoch, summary_writer)

        # evaluate on validation set
        prec1 = validate(model, val_loader, criterion, args.gpu, epoch, summary_writer)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save_dir)

    summary_writer.close()


if __name__ == '__main__':
    main()
