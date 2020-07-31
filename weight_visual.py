
import torchvision.models as models
import warnings
import random
import numpy as np
import os
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils.train_val import train, save_checkpoint, validate
from utils.data_loader import load_train_data, load_val_data
from utils.get_cifar_data import cifar_train_data, cifar_val_data
from quantize import quantize_guided
from quantize.quantize_method import quantize_weights_bias_gemm
from net import net_quantize_activation, net_quantize_weight, net_bn_conv_merge, net_bn_conv_merge_quantize
from tensorboardX import SummaryWriter

from vgg import vgg, quan_w_vgg, qa_vgg, vgg_guided, vgg_new
from scale.scale_quantize_method import quantize_weight_gemm,quantize_bias_gemm,quantize_activations_gemm
from resnet import resnet


def def_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path,i)
        if os.path.isdir(c_path):
            def_file(c_path)
        else:
            os.remove(c_path)

save_dir = 'weight_visdom/weight0'
if os.path.isdir(save_dir):
    def_file(save_dir)
else:
    os.mkdir(save_dir)

gpu = [0]
epsilon = 1e-5

model = resnet.ResNet18()
merge_model = net_bn_conv_merge.resnet18()
# merge_model = net_bn_conv_merge_quantize.resnet18()
# model = vgg.vgg16_bn()
# merge_model = vgg.vgg16()
# print(model)
# print(merge_model)

model = torch.nn.DataParallel(model, gpu).cuda()
merge_model = torch.nn.DataParallel(merge_model, gpu).cuda()

init_model = torch.load('resnet/resnet18_cifar100/model_best.pth.tar')
model.load_state_dict(init_model['state_dict'])
print(init_model['best_prec1'])

state_dict = model.state_dict()
merge_state_dict = merge_model.state_dict()


# for k in state_dict:
#      print(k)

# for k in merge_state_dict:
#     print(k)

weight = state_dict["module.linear.weight"]
bias = state_dict["module.linear.bias"]
qweight = quantize_weights_bias_gemm(state_dict["module.linear.weight"])
qbias = quantize_weights_bias_gemm(state_dict["module.linear.bias"])
merge_state_dict.update({"module.linear.weight": qweight,
                        "module.linear.bias": qbias})
del state_dict["module.linear.weight"]
del state_dict["module.linear.bias"]

params = np.array(list(state_dict.keys()))
params = params.reshape((-1, 6))
# params = params.reshape((-1, 6))
l_vgg = ['.0.','.2.','.5.', '.7.', '.10.', '.12.', '.14.', '.17.',
         '.19.', '.21.', '.24.', '.26.', '.28.']
scale =[]
for index in range(params.shape[0]):
    weight = state_dict[params[index][0]]
    gamma = state_dict[params[index][1]]
    beta = state_dict[params[index][2]]
    running_mean = state_dict[params[index][3]]
    running_var = state_dict[params[index][4]]
    delta = gamma/(torch.sqrt(running_var+epsilon))
    weight = weight * delta.view(-1, 1, 1, 1)
    bias = (0-running_mean) * delta + beta
    qweight,s = quantize_weight_gemm(weight)
    qbias = quantize_bias_gemm(bias)
    scale.append(s)
    merge_state_dict.update({params[index][0]: qweight,
                             params[index][0][:-6] + "bias": qbias})
#    merge_state_dict.update({params[index][0][0:15] + l_vgg[index] + "weight": weight,
#                             params[index][0][0:15] + l_vgg[index] + "bias":  bias})

merge_model.load_state_dict(merge_state_dict)

print(scale)
summary_writer = SummaryWriter(save_dir)
x = state_dict["module.conv1.weight"]
y = merge_state_dict["module.conv1.weight"]
z = y*scale[1]
print('x:')
print(torch.max(x) - torch.min(x))
print(torch.mean(x))
print(torch.var(x))
print('y:')
print(torch.max(y) - torch.min(y))
print(torch.mean(y))
print(torch.var(y))
print('z:')
print(torch.max(z) - torch.min(z))
print(torch.mean(z))
print(torch.var(z))

v = x.view(-1)
w = y.view(-1)
u = z.view(-1)

len = v.size()
print(len)
len = 1728
print(len)
for i in range(len):
    summary_writer.add_scalar('x_scalar', v[i], i)
    summary_writer.add_scalar('y_scalar', w[i], i)
    summary_writer.add_scalar('z_scale', u[i], i)

summary_writer.add_histogram('x_hist', v)
summary_writer.add_histogram('y_hist', w)
summary_writer.add_histogram('z_hist', u)
summary_writer.close()

