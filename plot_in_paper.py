import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import lpips
import inversefed

# datafile = 'resultsResNet50weight1mixupmax1000noise0_2alpha0_7dropoutFalsetrainFalsebatchFalse'
# datafile = 'resultsResNet18weight1reproducemax501noise0_2alpha1_0dropoutFalsetrainFalsebatchFalse'
datafile = 'resultsResNet18CIFAR100500weight20cifar100max500noise0_2alpha1_0convTruewgt1formerf_wgt0'
image_batch = torch.load("./"+datafile+"/images/batch0.pt", map_location="cpu")
image_batch_org = torch.load("./"+datafile+"/images/batch0_org.pt", map_location="cpu")
image_batch_forg = torch.load("./"+datafile+"/images/batch0_forg.pt", map_location="cpu")
label = torch.load('./'+datafile+'/label.pt',map_location="cpu") 

plot_idx = []
offset = 1
for i in range(10):
    plot_idx.append((label[offset:] == i).nonzero()[0].item() + offset)
import ipdb; ipdb.set_trace()
tt = transforms.ToPILImage()


# image_batch = torch.load("./attack_adaptive1.pt", map_location="cpu")
# plt.figure(figsize=(12, 8))
# for i in range(128):
#     plt.subplot(8, 16, i + 1)
#     plt.imshow(tt(image_batch[i].squeeze()))
#     # plt.title("image=%d" % (i))
#     plt.axis('off')

# plt.figure(figsize=(12, 8))
# for i in range(128):
#     plt.subplot(8, 16, i + 1)
#     plt.imshow(tt(image_batch_org[i].squeeze()))
#     # plt.title("image=%d" % (i))
#     plt.axis('off')

# plt.figure(figsize=(12, 8))
# for i in range(128):
#     plt.subplot(8, 16, i + 1)
#     plt.imshow(tt(image_batch_forg[i].squeeze()))
#     # plt.title("image=%d" % (i))
#     plt.axis('off')

plt.figure(figsize=(10, 3))
for i in range(10):
    plt.subplot(3, 10, i + 1)
    plt.imshow(tt(image_batch[plot_idx[i]].squeeze()))
    plt.axis('off')
    plt.subplot(3, 10, i + 11)
    plt.imshow(tt(image_batch_forg[plot_idx[i]].squeeze()))
    plt.axis('off')
    plt.subplot(3, 10, i + 21)
    plt.imshow(tt(image_batch_org[plot_idx[i]].squeeze()))
    # plt.title("image=%d" % (i))
    plt.axis('off')

plt.show()

percept_loss_fn = lpips.LPIPS(net='vgg')
# d_rand = percept_loss_fn.forward(gen, public_x).squeeze()
d_feat_s = percept_loss_fn.forward(image_batch, image_batch_org).squeeze()
d_feat_t = percept_loss_fn.forward(image_batch, image_batch_forg).squeeze()
print("lpips with starting mean: {:2.3f}\tstd: {:2.3f} ".format(d_feat_s.mean().item(), d_feat_s.std().item()))
print("lpips with target mean: {:2.3f}\tstd: {:2.3f} ".format(d_feat_t.mean().item(), d_feat_t.std().item()))

psnr_s = inversefed.metrics.psnr(image_batch, image_batch_org)
psnr_t = inversefed.metrics.psnr(image_batch, image_batch_forg)
print(f'PSNR with starting: {psnr_s:2.2f}')
print(f"PSNR with target: {psnr_t:2.2f}")

