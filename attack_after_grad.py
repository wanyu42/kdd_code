import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import resnet
import lpips
from data_loader import get_cifar10_loader
from torchvision import transforms, datasets
import os
import inversefed
from grad_prune import GradPruneDefense
import argparse

# Performs the gradient leakage attack on the obscured data, 
# Compare leaked data with private data,
# 

setup = inversefed.utils.system_startup()

def parse_args():
    parser = argparse.ArgumentParser(description='gradattack training')
    parser.add_argument('--cost_fn', default='sim', type=str)
    parser.add_argument('--tv_weight', default=0.0, type=float)
    parser.add_argument('--bn_reg', default=0.0, type=float)
    parser.add_argument('--attack_images', default=20, type=int)
    parser.add_argument('--attack_iter', default=5000, type=int)
    parser.add_argument('--attack_path', default=None, type=str)
    args = parser.parse_args()

    return args

arch = 'ResNet18'
# # defense = 'None'
# cost_fn = 'sim_cmpr0.5'
# num_images = 1
# # attack_images = 1
# attack_images = 50
# batch_size = 128
# max_attack_iterations = 5000
# # max_attack_iterations = 4
# # grad_prune_rate = 0.9
# bn_reg = 0
# # bn_reg = 0.001
# tv_weight = 1e-4
# # tv_weight = 1e-6

args = parse_args()
cost_fn = args.cost_fn
num_images = 1
attack_images = args.attack_images
batch_size = 128
max_attack_iterations = args.attack_iter
bn_reg = args.bn_reg
tv_weight = args.tv_weight
resultpath = args.attack_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_fn = torch.nn.CrossEntropyLoss()
percept_loss_fn = lpips.LPIPS(net='vgg')
if torch.cuda.is_available():
        percept_loss_fn = percept_loss_fn.cuda()

model = resnet.ResNet18()
model.to(**setup)
model.eval()
# model.train()

# defense_method = GradPruneDefense(grad_prune_rate)

dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
def plot(tensor):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    if tensor.shape[0] == 1:
        return plt.imshow(tensor[0].permute(1, 2, 0).cpu())
    else:
        fig, axes = plt.subplots(1, tensor.shape[0], figsize=(12, tensor.shape[0]*12))
        for i, im in enumerate(tensor):
            axes[i].imshow(im.permute(1, 2, 0).cpu())


# resultpath = 'resultsResNet18weight1reproducemax501noise0_2alpha1_0dropoutFalsetrainFalsebatchFalse'
# attackpath = 'paper'+cost_fn+'tv'+str(tv_weight)+resultpath
attackpath = 'svhn'+cost_fn+'tv'+str(tv_weight)+resultpath
# attackpath = 'eval_attack_defense'+cost_fn+'tv'+str(tv_weight)
# attackpath = 'attack_defense'
generated = torch.load(resultpath+'/images/batch0.pt', map_location='cpu').to(device)
private_x = torch.load(resultpath+'/images/batch0_forg.pt', map_location='cpu').to(setup['device'])
labels = torch.load(resultpath+'/label.pt', map_location='cpu').to(setup['device'])
plot_idx = []
offset = 1
for i in range(10):
    plot_idx.append((labels[offset:] == i).nonzero()[0].item() + offset)
    plot_idx.append((labels[offset:] == i).nonzero()[1].item() + offset)
    

config = dict(signed=True,
              boxed=True,
            #   cost_fn='sim',
              cost_fn=cost_fn,
              indices='def',
            #   indices='top50',
              weights='equal',
              lr=0.1,
              optim='adam',
              restarts=1,
              max_iterations=max_attack_iterations,
            #   total_variation=1e-6,
              total_variation=tv_weight,
              init='randn',
            #   init='zeros',
              filter='none',
              lr_decay=True,
              scoring_choice='loss',
              first_bn_multiplier=1,
              bn_reg=bn_reg,
              )

g_loss = []
g_psnr = []
g_lpips = []
g_mse = []
g_fmse = []

attack_outputs = []
ground_truth_list = []
private_list = []

if not os.path.isdir(attackpath):
    os.mkdir(attackpath)

for i in plot_idx:

    # dst = datasets.CIFAR10("~/dataset", download=True)
    # tp = transforms.ToTensor()
    # tt = transforms.ToPILImage()
    # img_index=0
    # gt_data = tp(dst[img_index][0]).to(device)
    # # gt_data = gt_data.view(1, *gt_data.size())
    # gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
    # gt_label = gt_label.view(1, )
    # # gt_onehot_label = label_to_onehot(gt_label)
    # ground_truth = gt_data.sub(dm).div(ds).unsqueeze(0).contiguous().to(setup['device'])
    # ======================================================================================

    # ground_truth = org[i].sub(dm).div(ds).unsqueeze(0).contiguous().to(setup['device'])
    ground_truth = generated[i].sub(dm).div(ds).unsqueeze(0).contiguous().to(setup['device'])
    private_truth = private_x[i].sub(dm).div(ds).unsqueeze(0).contiguous().to(setup['device'])
    label = labels[i].view((-1,)).to(setup['device'])
    
    plot(ground_truth)
    plt.savefig(attackpath+'/attack_obscured'+str(i)+'.jpg')
    plot(private_truth)
    plt.savefig(attackpath+'/attack_private'+str(i)+'.jpg')

    model.zero_grad()

    # target_loss, _, _ = loss_fn(model(ground_truth), labels)
    target_loss = loss_fn(model(ground_truth), label)
    input_gradient = torch.autograd.grad(target_loss, model.parameters())
    # import ipdb; ipdb.set_trace()
    # if defense != 'None':
    #     input_gradient = defense_method.apply(input_gradient)
    input_gradient = [grad.detach() for grad in input_gradient]

    # Reconstruct
    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=num_images)
    output, stats = rec_machine.reconstruct(input_gradient, label, img_shape=(3, 32, 32))

    # Calculate Stat
    # test_mse = (output.detach() - ground_truth).pow(2).mean()
    # feat_mse = (model(output.detach())- model(ground_truth)).pow(2).mean()  
    # test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1/ds)
    # test_lpips = percept_loss_fn.forward(output.detach(), ground_truth.detach()).squeeze()
    test_mse = (output.detach() - private_truth).pow(2).mean()
    feat_mse = (model(output.detach())- model(private_truth)).pow(2).mean()  
    test_psnr = inversefed.metrics.psnr(output, private_truth, factor=1/ds)
    test_lpips = percept_loss_fn.forward(output.detach(), private_truth.detach()).squeeze()

    g_loss.append(stats['opt'])
    g_mse.append(test_mse.item())
    g_fmse.append(feat_mse.item())
    g_psnr.append(test_psnr)
    g_lpips.append(test_lpips.item())

    attack_outputs.append(output.detach())
    ground_truth_list.append(ground_truth.detach())
    private_list.append(private_truth.detach())

    # Save outputs
    plot(output)
    plt.title(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
            f"| PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} | LPIPS: {test_lpips:2.4f} |")
    # if not os.path.isdir('./attack_test_results'):
    #     os.mkdir('./attack_test_results')
    # plt.savefig('./attack_test_results/'+'attack_'+str(i)+'.jpg')
    plt.savefig(attackpath+'/attack_output'+str(i)+'.jpg')

attack_outputs = torch.stack(attack_outputs)
torch.save(attack_outputs, attackpath+'/attack_output.pt')
ground_truth_list = torch.stack(ground_truth_list)
torch.save(ground_truth_list, attackpath+'/obscured.pt')
private_list = torch.stack(private_list)
torch.save(private_list, attackpath+'/private.pt')

# import ipdb;ipdb.set_trace()
g_loss = np.array(g_loss)
g_mse = np.array(g_mse)
g_fmse = np.array(g_fmse)
g_psnr = np.array(g_psnr)
g_lpips = np.array(g_lpips)
print('==========================================================')
print(f"Rec.loss: {g_loss.mean():2.4f} +- {g_loss.std():2.4f}")
print(f"MSE: {g_mse.mean():2.4f} +- {g_mse.std():2.4f}")
print(f"F_MSE: {g_fmse.mean():2.4e} +- {g_fmse.std():2.4e}")
print(f"PSNR: {g_psnr.mean().item():4.2f} +- {g_psnr.std().item():4.2f}\tmax: {g_psnr.max():4.2f}")
print(f"LPIPS: {g_lpips.mean().item():4.3f} +- {g_lpips.std().item():4.3f}\tmin: {g_lpips.min():4.3f}")
print('==========================================================')
print('Defense: '+cost_fn+'\tBN_Reg: '+str(bn_reg)+'\tTV: '+str(tv_weight))
print(resultpath)
