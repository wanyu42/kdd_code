import os
import time
import numpy as np
import lpips 
import inversefed
import torch
from torchvision import transforms

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

weight_range = [10, 20, 30, 40, 50, 60]
# weight_range = [40,60]
# noise 0.2 means pretrained models, 0.0 means random initialized models
noise_value = 0.2
starting = "cifar100"
alpha_value = 1.0
conv = True
weight_conv = 1
conv_part = 'former'
# conv_part = 'whole'
defense = 'None'
max_iter = 500
model_type = "ResNet18CIFAR100"
feat_weight = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

percept_loss_fn = lpips.LPIPS(net='vgg')
if torch.cuda.is_available():
    percept_loss_fn = percept_loss_fn.cuda()

for weight_value in weight_range:
    datafile = 'results'+model_type+str(max_iter)+"weight"+ str(weight_value).replace(".", "_")+starting+"max"+str(max_iter)+"noise" \
                    +str(noise_value).replace(".", "_")+"alpha"+str(alpha_value).replace(".", "_") \
                    +"conv"+str(conv)+"wgt"+str(weight_conv).replace(".", "_")+conv_part+"f_wgt"+str(feat_weight)

    generated = torch.load("./"+datafile+"/images/batch0.pt", map_location=device)[:20]
    public_x = torch.load("./"+datafile+"/images/batch0_org.pt", map_location=device)[:20]
    private_x = torch.load("./"+datafile+"/images/batch0_forg.pt", map_location=device)[:20]
    tt = transforms.ToPILImage()
    # import ipdb; ipdb.set_trace()

    print('==========================================================')
    print(datafile)
    d_feat = percept_loss_fn.forward(private_x, generated).squeeze()
    print(f"lpips: {d_feat.mean().item():4.3f} +- {d_feat.std().item():4.3f}\tmin: {d_feat.min():4.3f}")

    d_psnr = torch.tensor([psnr(private_x[i].detach().unsqueeze(0), \
                            generated[i].unsqueeze(0)) for i in range(private_x.shape[0])])
    print(f"psnr: {d_psnr.mean().item():4.2f} +- {d_psnr.std().item():4.2f}\tmax: {d_psnr.max().item():4.2f}")
    print('==========================================================')