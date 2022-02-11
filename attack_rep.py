import os
import torch
import torch.nn as nn
import torch.nn.functional as F #233
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from data_loader import GenDataset, CompareDataset, CIFAR10


alpha = 0.2
# data_dict = './resultsResNet50weight1mixupmax1000noise0_2alpha'+str(alpha).replace('.', '_')+'dropoutFalsetrainFalsebatchFalse/'
# data_dict = './resultsResNet18weight1reproducemax501noise0_2alpha1_0dropoutFalsetrainFalsebatchFalse'
data_dict = './attack_results_batch128'

tt = transforms.ToPILImage()

# train_loader = torch.utils.data.DataLoader(
#                  CompareDataset(data_dict, transform=transforms.ToTensor()),
#                  batch_size= 50, shuffle=False)
# public_dataset = CIFAR10(50, 0.0, mode = "valid", transform=transforms.ToTensor())
# public_loader = torch.utils.data.DataLoader(public_dataset,batch_size= 500, shuffle=False)
# public_x = list(public_loader)[0][0]
# public_y = list(public_loader)[0][1]

gen = torch.load(data_dict+'/images/batch0.pt', map_location='cpu')
public_x = torch.load(data_dict+'/images/batch0_org.pt', map_location='cpu')
reconstruct = torch.load(data_dict+'/attack_outs.pt', map_location='cpu')

# for idx, (gen, org, target) in enumerate(train_loader):
gen_d = gen.view(gen.shape[0], -1)
public_d = public_x.view(public_x.shape[0], -1)

inner_p = (gen_d/torch.norm(gen_d, dim=1).view(-1,1)) @ (public_d/torch.norm(public_d, dim=1).view(-1,1)).T
# idx_pub = torch.argmax(inner_p, dim=1)
max_pub, idx_pub = torch.max(inner_p, dim = 1)

# import ipdb; ipdb.set_trace()
# reconstructed = (gen - alpha*public_x[idx_pub])/(1-alpha)
# reconstructed = (gen - max_pub.view(-1,1,1,1) * public_x[idx_pub]) / (1-max_pub.view(-1,1,1,1))

import lpips
percept_loss_fn = lpips.LPIPS(net='alex')
# d_rand = percept_loss_fn.forward(gen, public_x).squeeze()
# d_align = percept_loss_fn.forward(gen, public_x[idx_pub]).squeeze()
d_reconstruct = percept_loss_fn.forward(reconstruct, public_x[idx_pub[1]])
print(d_reconstruct)
# print("d_rand: ")
# print(d_rand)
# print("d_align: ")
# print(d_align)

import matplotlib.pyplot as plt
plt.figure()
# import ipdb; ipdb.set_trace()
plt.subplot(1, 3, 1)
plt.imshow(tt(reconstruct.squeeze()))
plt.subplot(1, 3, 2)
plt.imshow(tt(public_x[idx_pub[1]].squeeze()))
plt.subplot(1, 3, 3)
plt.imshow(tt(gen[1].squeeze()))
plt.show()

if not os.path.isdir('./figures_'+data_dict.split('/')[1]+'/'):
    os.mkdir('./figures_'+data_dict.split('/')[1]+'/')
(tt(reconstruct.squeeze())).save('./figures_'+data_dict.split('/')[1]+'/'+"reconstruct"+'.jpg')
(tt(public_x[idx_pub[0]].squeeze())).save('./figures_'+data_dict.split('/')[1]+'/'+"org"+'.jpg')
(tt(gen[0].squeeze())).save('./figures_'+data_dict.split('/')[1]+'/'+"gen"+'.jpg')
# for i in range(gen.shape[0]):
#     # (tt(reconstructed[i].squeeze())).save('./figures_'+data_dict.split('/')[1]+'/'+ str(i) +"rec"+'.jpg')
#     # (tt(org[i].squeeze())).save('./figures_'+data_dict.split('/')[1]+'/'+str(i)+"org"+'.jpg')
#     # (tt(gen[i].squeeze())).save('./figures_'+data_dict.split('/')[1]+'/'+str(i)+"gen"+'.jpg')
#     # (tt(public_x[idx_pub[i]].squeeze())).save('./figures_'+data_dict.split('/')[1]+'/'+str(i)+"guess_pub"+'.jpg')
#     (tt(gen[i].squeeze())).save('./figures_'+data_dict.split('/')[1]+'/'+str(i)+"gen"+'.jpg')
#     (tt(public_x[idx_pub[i]].squeeze())).save('./figures_'+data_dict.split('/')[1]+'/'+str(i)+"org"+'.jpg')
# # import ipdb; ipdb.set_trace()
    

