import os
import time
import numpy as np
import lpips 
import inversefed
import torch
from torchvision import transforms

def normalize_batch(img):
    img -= torch.amin(img, dim=(1, 2, 3), keepdims=True)
    img /= torch.amax(img, dim=(1, 2, 3), keepdims=True)

    # img *= 2
    # img -= 1
    return img

def psnr(img_batch, ref_batch, batched=False, factor=1.0):
    """Standard PSNR."""
    def get_psnr(img_in, img_ref):
        mse = ((img_in - img_ref)**2).mean()
        if mse > 0 and torch.isfinite(mse):
            return (10 * torch.log10(factor**2 / mse))
        elif not torch.isfinite(mse):
            return img_batch.new_tensor(float('nan'))
        else:
            return img_batch.new_tensor(float('inf'))

    if batched:
        psnr = get_psnr(img_batch.detach(), ref_batch)
    else:
        [B, C, m, n] = img_batch.shape
        psnrs = []
        for sample in range(B):
            psnrs.append(get_psnr(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
        psnr = torch.stack(psnrs, dim=0).mean()

    return psnr.item()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
percept_loss_fn = lpips.LPIPS(net='vgg')
if torch.cuda.is_available():
    percept_loss_fn = percept_loss_fn.cuda()

cost_fn = 'sim'
tv_weight = 1e-2
# weight_range = [10, 20, 30, 40, 50, 60]
weight_range = [1]

for weight in weight_range:
    resultpath = 'resultsResNet18SVHN500weight1cifar100max500noise0_2alpha1_0convTruewgt1latterf_wgt1'
    # resultpath = 'resultsResNet18CIFAR100500weight'+str(weight)+'cifar100max500noise0_2alpha1_0convTruewgt1formerf_wgt0'
    # datafile = 'attack'+cost_fn+'tv'+str(tv_weight)+resultpath
    datafile = 'svhn'+cost_fn+'tv'+str(tv_weight)+resultpath

    attack_output = normalize_batch(torch.load("./"+datafile+"/attack_output.pt", map_location=device).squeeze())
    # obscured = normalize_batch(torch.load("./"+datafile+"/obscured.pt", map_location=device).squeeze())
    # private_x = normalize_batch(torch.load("./"+datafile+"/private.pt", map_location=device).squeeze())
    obscured = torch.load(resultpath+'/images/batch0.pt', map_location='cpu').to(device)[:20]
    private_x = torch.load(resultpath+'/images/batch0_forg.pt', map_location='cpu').to(device)[:20]
    tt = transforms.ToPILImage()
    # import ipdb; ipdb.set_trace()

    print('==========================================================')
    print(datafile)
    d_feat = percept_loss_fn.forward(private_x, attack_output).squeeze()
    print(f"lpips (private, attackout): {d_feat.mean().item():4.3f} +- {d_feat.std().item():4.3f}\tmin: {d_feat.min():4.3f}")

    d_psnr = torch.tensor([psnr(private_x[i].detach().unsqueeze(0), \
                            attack_output[i].unsqueeze(0)) for i in range(private_x.shape[0])])
    print(f"psnr (private, attackout): {d_psnr.mean().item():4.2f} +- {d_psnr.std().item():4.2f}\tmax: {d_psnr.max().item():4.2f}")
    print('==========================================================')
    # print(datafile)
    d_feat = percept_loss_fn.forward(private_x, obscured).squeeze()
    print(f"lpips (private, obscured): {d_feat.mean().item():4.3f} +- {d_feat.std().item():4.3f}\tmin: {d_feat.min():4.3f}")

    d_psnr = torch.tensor([psnr(private_x[i].detach().unsqueeze(0), \
                            obscured[i].unsqueeze(0)) for i in range(private_x.shape[0])])
    print(f"psnr (private, obscured): {d_psnr.mean().item():4.2f} +- {d_psnr.std().item():4.2f}\tmax: {d_psnr.max().item():4.2f}")
    print('==========================================================')
    # print(datafile)
    # d_feat = percept_loss_fn.forward(attack_output, obscured).squeeze()
    # print(f"lpips (attackout, obscured): {d_feat.mean().item():4.3f} +- {d_feat.std().item():4.3f}\tmin: {d_feat.min():4.3f}")

    # d_psnr = torch.tensor([psnr(attack_output[i].detach().unsqueeze(0), \
    #                         obscured[i].unsqueeze(0)) for i in range(attack_output.shape[0])])
    # print(f"psnr (attackout, obscured): {d_psnr.mean().item():4.2f} +- {d_psnr.std().item():4.2f}\tmax: {d_psnr.max().item():4.2f}")
    # print('==========================================================')