import os
import torch
import torch.nn as nn
import torch.nn.functional as F #233
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from data_loader import GenDataset, CompareDataset, CIFAR10
import inversefed
import lpips
from dataset_gen_new import pgd_step, gd_step, loss_feature_diff, ConvOutHook
import resnet
import matplotlib.pyplot as plt
import lpips
from inversefed.metrics import total_variation as TV


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
percept_loss_fn = lpips.LPIPS(net='vgg')
if torch.cuda.is_available():
    percept_loss_fn = percept_loss_fn.cuda()

def normalize_batch(img):
    img -= torch.amin(img, dim=(1, 2, 3), keepdims=True)
    img /= torch.amax(img, dim=(1, 2, 3), keepdims=True)
    return img


# alpha = 0.2
# data_dict = './resultsResNet50weight1mixupmax1000noise0_2alpha'+str(alpha).replace('.', '_')+'dropoutFalsetrainFalsebatchFalse/'
# data_dict = './resultsResNet18weight1reproducemax501noise0_2alpha1_0dropoutFalsetrainFalsebatchFalse'
# data_dict = './attack_results_batch128'
data_dict = './resultsResNet18CIFAR100500weight20cifar100max500noise0_2alpha1_0convTruewgt1formerf_wgt0'

tt = transforms.ToPILImage()

gen = torch.load(data_dict+'/images/batch0.pt', map_location=device)
public_x = torch.load(data_dict+'/images/batch0_org.pt', map_location=device)
private_x = torch.load(data_dict+'/images/batch0_forg.pt', map_location=device)
label = torch.load(data_dict+'/label.pt', map_location=device)
# reconstruct = torch.load(data_dict+'/attack_outs.pt', map_location='cpu')

plot_idx = []
offset = 1
for i in range(10):
    plot_idx.append((label[offset:] == i).nonzero()[0].item() + offset)
plot_idx = torch.tensor(plot_idx)
gen = gen[plot_idx]
public_x = public_x[plot_idx]
private_x = private_x[plot_idx]

# data_dict = './papersimtv0.0001resultsResNet18CIFAR100500weight20cifar100max500noise0_2alpha1_0convTruewgt1formerf_wgt0'
data_dict = './attacksimtv0.0001resultsResNet18CIFAR100500weight20cifar100max500noise0_2alpha1_0convTruewgt1formerf_wgt0'
# data_dict = './svhnsimtv0.01resultsResNet18SVHN500weight1cifar100max500noise0_2alpha1_0convTruewgt1latterf_wgt1'
setup = inversefed.utils.system_startup()
dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
gen = torch.load(data_dict+'/attack_output.pt', map_location=device).squeeze().mul_(ds).add_(dm).clamp_(0, 1)
private_x = torch.load(data_dict+'/private.pt', map_location=device).squeeze().mul_(ds).add_(dm).clamp_(0, 1)

# import ipdb; ipdb.set_trace()

modelpath = './trained_models/best_CIFAR100_ResNet18None_noise0.0_alpha0.0_results.pt'
inner_per_image = 100
inner_per_distance_record = 10
batch_size = 128

model = resnet.ResNet18(num_classes=100)
model.load_state_dict(torch.load(modelpath, map_location=device))
model = model.to(device)
model.eval()

params = {
        "weight" : 5,
        "weight_conv" : 1.0,
        "noise_level" : 0.2,
        "start_point" : "cifar100",
        "alpha" : 1.0,
        "step_size" : 0.1,
        "model_type" : "ResNet18CIFAR100",
        "max_iterations": 500,
        "result" : '',
        "defense" : 'None',
        "swap_num" : 128,
        "conv_part" : 'former',
        "loss_type" : "l2", 
        "conv":True, 
        "drop_random":False,
        "pgd":True,
        "feat_weight": 0,
    }
conv_out_layers = []
if params['conv']==True:
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            conv_out_layers.append(ConvOutHook(module))
data_r = gen.detach().clone()
# data_r.requires_grad = True
# data = torch.rand(data_r.shape, device=device)
data = 0.5*torch.ones(data_r.shape, device=device)
# data = public_x.detach().clone()

for step_num in range(100):
    data.requires_grad = True

    total_vjp = 0
    for i in range(5):
        def fun(x):
            output_target = model(x)
            conv_out = [mod.output for mod in conv_out_layers]
            return conv_out[i]
        v = fun(data_r) - fun(data)
        _, vjp = torch.autograd.functional.vjp(fun, data_r, v, create_graph=True)
        total_vjp += vjp

    loss = torch.norm(total_vjp - params['weight']*(data_r-data)) #+ 1 * TV(data)
    grad = torch.autograd.grad(loss, data)[0]
    data = data.detach() - 0.01 * grad
    data = torch.clamp(data, min=0.0, max=1.0)
    if step_num % 100==0:
        print(step_num)
        print(f'loss: {loss.item():4.4f}')
##====================================================================================
data = torch.clamp(data, 0.0, 1.0)
torch.save(data,'./attack_grad_adaptive2_3.pt')

# feat_mse = (model(output.detach())- model(ground_truth)).pow(2).mean()  
# import ipdb; ipdb.set_trace()
test_psnr = torch.tensor([inversefed.metrics.psnr(data[i].detach().unsqueeze(0), \
    private_x[i].unsqueeze(0)) for i in range(private_x.shape[0])])
test_lpips = percept_loss_fn.forward(data.detach(), private_x.detach()).squeeze()
print('==========================================================')
# print(f"MSE: {g_mse.mean():2.4f} +- {g_mse.std():2.4f}")
print(f"PSNR: {test_psnr.mean():4.2f} +- {test_psnr.std():4.2f}")
print(f"LPIPS: {test_lpips.mean():2.4f} +- {test_lpips.std():2.4f}")
print('==========================================================')
# print(f"MSE: {g_mse.min():2.4f}")
print(f"PSNR: {test_psnr.max():4.2f}")
print(f"LPIPS: {test_lpips.min():2.4f}")

# plt.figure(figsize=(12, 8))
# for i in range(128):
#     plt.subplot(8, 16, i + 1)
#     plt.imshow(tt(data_r[i].squeeze()))
#     # plt.title("image=%d" % (i))
#     plt.axis('off')