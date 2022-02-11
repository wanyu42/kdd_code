import os
import time
import numpy as np
import lpips 
import inversefed
import torch
from torchvision import transforms
# from metric_after_grad import normalize_batch

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
tv_weight_range = [1e-4, 1e-3, 1e-2, 1e-1]

for tv_weight in tv_weight_range:
    # datafile = 'eval_attack_defense'+cost_fn+'tv'+str(tv_weight)
    datafile = 'svhn_attack_defense'+cost_fn+'tv'+str(tv_weight)

    generated = torch.load("./"+datafile+"/attack_output.pt", map_location=device).squeeze()
    private_x = torch.load("./"+datafile+"/ground_truth.pt", map_location=device).squeeze()
    tt = transforms.ToPILImage()
    # import ipdb; ipdb.set_trace()

    print('==========================================================')
    print(datafile)
    d_feat = percept_loss_fn.forward(private_x, generated).squeeze()
    print(f"Attack lpips: {d_feat.mean().item():4.3f} +- {d_feat.std().item():4.3f}\tmin: {d_feat.min():4.3f}")

    d_psnr = torch.tensor([psnr(private_x[i].detach().unsqueeze(0), \
                            generated[i].unsqueeze(0)) for i in range(private_x.shape[0])])
    print(f"Attack psnr: {d_psnr.mean().item():4.2f} +- {d_psnr.std().item():4.2f}\tmax: {d_psnr.max().item():4.2f}")
    print('==========================================================')