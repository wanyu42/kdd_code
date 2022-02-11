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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
percept_loss_fn = lpips.LPIPS(net='vgg')
if torch.cuda.is_available():
    percept_loss_fn = percept_loss_fn.cuda()

def normalize_batch(img):
    img -= torch.amin(img, dim=(1, 2, 3), keepdims=True)
    img /= torch.amax(img, dim=(1, 2, 3), keepdims=True)
    return img


# alpha = 0.2
# # data_dict = './resultsResNet50weight1mixupmax1000noise0_2alpha'+str(alpha).replace('.', '_')+'dropoutFalsetrainFalsebatchFalse/'
# # data_dict = './resultsResNet18weight1reproducemax501noise0_2alpha1_0dropoutFalsetrainFalsebatchFalse'
# # data_dict = './attack_results_batch128'
# data_dict = './resultsResNet18CIFAR100500weight20cifar100max500noise0_2alpha1_0convTruewgt1formerf_wgt0'

# tt = transforms.ToPILImage()

# gen = torch.load(data_dict+'/images/batch0.pt', map_location=device)
# public_x = torch.load(data_dict+'/images/batch0_org.pt', map_location=device)
# private_x = torch.load(data_dict+'/images/batch0_forg.pt', map_location=device)
# # reconstruct = torch.load(data_dict+'/attack_outs.pt', map_location='cpu')

# gen_d = gen.view(gen.shape[0], -1)
# public_d = public_x.view(public_x.shape[0], -1)

# inner_p = (gen_d/torch.norm(gen_d, dim=1).view(-1,1)) @ (public_d/torch.norm(public_d, dim=1).view(-1,1)).T
# alpha = torch.diag(inner_p)

# reconstructed = normalize_batch(gen - alpha.view(-1,1,1,1)*public_x)
# # reconstructed = normalize_batch( ((gen_d/torch.norm(gen_d, dim=1).view(-1,1)) - \
# #     alpha.view(-1,1)*(public_d/torch.norm(public_d, dim=1).view(-1,1))).view(-1, 3, 32, 32) )
# # import ipdb; ipdb.set_trace()
# # reconstructed = (gen - alpha*public_x[idx_pub])/(1-alpha)

# d_reconstruct = percept_loss_fn.forward(reconstructed, private_x)
# d_encode = percept_loss_fn.forward(gen, private_x)
# print(f"Before Attack LPIPS: {d_encode.mean():2.4f} +- {d_encode.std():2.4f}")
# print(f"After Attack LPIPS: {d_reconstruct.mean():2.4f} +- {d_reconstruct.std():2.4f}")

##====================================================================================
params = {
        "weight" : 30,
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
        "conv_part" : 'whole',
        "loss_type" : "l2", 
        "conv":True, 
        "drop_random":False,
        "pgd":True,
        "feat_weight": 0,
    }

modelpath = './trained_models/best_CIFAR100_ResNet18None_noise0.0_alpha0.0_results.pt'
inner_per_image = 100
inner_per_distance_record = 10
batch_size = 128

model = resnet.ResNet18(num_classes=100)
model.load_state_dict(torch.load(modelpath, map_location=device))
model = model.to(device)
model.eval()

conv_out_layers = []
if params['conv']==True:
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            conv_out_layers.append(ConvOutHook(module))

data_dict = './resultsResNet18CIFAR100500weight20cifar100max500noise0_2alpha1_0convTruewgt1formerf_wgt0'
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

data = gen[plot_idx]
data_r = public_x[plot_idx]
private_x = private_x[plot_idx]
target = label[plot_idx]
# Test with grad leakage output
data_dict = './papersimtv0.0001resultsResNet18CIFAR100500weight20cifar100max500noise0_2alpha1_0convTruewgt1formerf_wgt0'
data_dict = './attacksimtv0.0001resultsResNet18CIFAR100500weight30cifar100max500noise0_2alpha1_0convTruewgt1wholef_wgt0'
setup = inversefed.utils.system_startup()
dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
data = torch.load(data_dict+'/attack_output.pt', map_location=device).squeeze().mul_(ds).add_(dm).clamp_(0, 1)
private_x = torch.load(data_dict+'/private.pt', map_location=device).squeeze().mul_(ds).add_(dm).clamp_(0, 1)
data_r = public_x[:private_x.shape[0]]
target = label[:private_x.shape[0]]

max_iterations = 100

data_r_init = data_r.detach().clone()
for inner in range(max_iterations):
    data_r.requires_grad = True

    feature_diff = loss_feature_diff(model, data, data_r, target, inner, conv_out_layers, \
        params['loss_type'], params['conv'], params['weight_conv'], params['conv_part'], params['feat_weight'])

    # image_diff = torch.sum((torch.clamp(data_r, min=0.0, max=1.0) - data)**2) / target.shape[0]
    image_diff = torch.sum((torch.clamp(data_r, min=0.0, max=1.0) - data_r_init)**2) / target.shape[0]
    loss_fn = feature_diff - params['weight'] * image_diff
    # import ipdb; ipdb.set_trace()
    if params['pgd']==True:
        data_r = pgd_step(loss_fn, data_r, target, params['step_size'])
    else:
        data_r = gd_step(loss_fn, data_r, target, params['step_size'])

    # if inner % inner_per_distance_record == 0:
    #     feature_diff_history.append(feature_diff.item())
    #     image_diff_history.append(image_diff.item())
    #     loss_history.append(loss_fn.item())

    if inner % inner_per_image == 0 or inner == max_iterations-1:
        print('Train Epoch:{}\tLoss:{:.6f}\tF_diff:{:.6f}\tI_diff:{:.3f}'.format(inner, loss_fn.item(), 
            feature_diff.item(), image_diff.item()))

data_r = torch.clamp(data_r, 0.0, 1.0)
torch.save(data_r,'./attack_our.pt')

test_mse = (data_r.detach() - private_x).pow(2).mean()
# feat_mse = (model(output.detach())- model(ground_truth)).pow(2).mean()  
# import ipdb; ipdb.set_trace()
test_psnr = torch.tensor([inversefed.metrics.psnr(data_r[i].detach().unsqueeze(0), \
    private_x[i].unsqueeze(0)) for i in range(private_x.shape[0])])
test_lpips = percept_loss_fn.forward(data_r.detach(), private_x.detach()).squeeze()
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