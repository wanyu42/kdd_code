import torch

#import deeprobust.image.netmodels.train_model as trainmodel
from torchvision import transforms
import argparse
from data_loader import *
from dataset_gen import *


# torch.manual_seed(0)
# torch.manual_seed(123)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnet_type = 'ResNet18'
# resnet_type = 'ResNet9'
# resnet_type = 'ResNet50'

originalDataPath="~/dataset"

if resnet_type == 'ResNet18':   
        modelpath = "./trained_models/CIFAR10_VAL_ResNet18_epoch_200.pt"
elif resnet_type == 'ResNet9':
        modelpath = './trained_models/last_CIFAR10_VAL_ResNet9_noise0.01_alpha0.0_results.pt'
elif resnet_type == 'ResNet50':
        modelpath = './trained_models/last_CIFAR10_VAL_ResNet50_noise0.0_alpha0.0_results.pt'

dropout_random = False  
batch_size = 128
resultpath = './attack_results_batch'+str(batch_size)+'/'

########################################################################
####################### Generate Dataset ###############################
########################################################################
# max_iterations = 501
# inner_per_image = 100
# inner_per_distance_record = 10


# transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         ])
# train_loader, _, test_loader = get_cifar10_loader(50, batch_size, transform_train)
# transform_valid = transforms.Compose([transforms.ToTensor(),])
# valid_dataset = CIFAR10(50, 0.0, mode = "valid", transform=transform_valid)
# ########################################################
# ##############  Load the trained model #################
# ########################################################
# if resnet_type == 'ResNet18':
#         model = resnet.ResNet18(random=dropout_random)
# elif resnet_type == 'ResNet9':
#         model = resnet.ResNet9(random=dropout_random)
# elif resnet_type == 'ResNet50':
#         model = resnet.ResNet50(random=dropout_random)

# checkpoint = torch.load(modelpath, map_location=device)
# model.load_state_dict(checkpoint)
# model = model.to(device)
# model.eval()

# if not os.path.exists(resultpath+'label.pt'):
#     generate_dataset(model, train_loader, valid_dataset, device,batch_size,resultpath, alpha = 1.0, start_point = 'reproduce', 
#             loss_type = 'l2', weight = 1, noise_level = 0.2,
#             max_iterations=max_iterations, inner_per_distance_record=inner_per_distance_record, inner_per_image=inner_per_image)
# else:
#     print('Already exists')

# ########################################################################
# ####################### Attack ################################
# ########################################################################
# print("================================================================")
# print("================================================================")

gen = torch.load(resultpath+'/images/batch0.pt', map_location='cpu').to(device)
org = torch.load(resultpath+'/images/batch0_org.pt', map_location='cpu').to(device)
label = torch.load(resultpath+'/label.pt', map_location='cpu').to(device)

import inversefed
import matplotlib.pyplot as plt
import lpips

dm = torch.as_tensor(inversefed.consts.cifar10_mean, device=device)[:, None, None]
ds = torch.as_tensor(inversefed.consts.cifar10_std, device=device)[:, None, None]
def plot(tensor):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    if tensor.shape[0] == 1:
        return plt.imshow(tensor[0].permute(1, 2, 0).cpu())
    else:
        fig, axes = plt.subplots(1, tensor.shape[0], figsize=(12, tensor.shape[0]*12))
        for i, im in enumerate(tensor):
            axes[i].imshow(im.permute(1, 2, 0).cpu())


per_attack_size = 1
attack_images = 20

g_loss = []
g_psnr = []
g_lpips = []
g_mse = []
g_fmse = []

config = dict(signed=True,
              boxed=True,
              cost_fn='sim',
              indices='def',
              weights='equal',
              lr=0.1,
              optim='adam',
              restarts=1,
              max_iterations=5000,
              total_variation=1e-6,
              init='randn',
              filter='none',
              lr_decay=True,
              scoring_choice='loss')
              
model_attack = resnet.ResNet18().to(device)
model_attack.eval()
percept_loss_fn = lpips.LPIPS(net='vgg')
if torch.cuda.is_available():
        percept_loss_fn = percept_loss_fn.cuda()

# Start the attack
# gen_attack = gen[0].sub(dm).div(ds).unsqueeze(0).contiguous().to(device)
for i in range(attack_images):
        gen_attack = gen[i].sub(dm).div(ds).unsqueeze(0).contiguous().to(device)
        label_attack = label[i].view((-1,)).to(device)
        plot(gen_attack)
        plt.savefig(resultpath+'/attack_org'+str(i)+'.jpg')

        model_attack.zero_grad()

        target_loss = F.cross_entropy(model_attack(gen_attack), label_attack)
        input_gradient = torch.autograd.grad(target_loss, model_attack.parameters())
        # import ipdb; ipdb.set_trace()
        input_gradient = [grad.detach() for grad in input_gradient]

        rec_machine = inversefed.GradientReconstructor(model_attack, (dm, ds), config, num_images=per_attack_size)
        output, stats = rec_machine.reconstruct(input_gradient, label_attack, img_shape=(3, 32, 32))

        torch.save(output, resultpath+'/attack_outs.pt')

        test_mse = (output.detach() - gen_attack).pow(2).mean()
        feat_mse = (model_attack(output.detach())- model_attack(gen_attack)).pow(2).mean()  
        test_psnr = inversefed.metrics.psnr(output, gen_attack, factor=1/ds)
        test_lpips = percept_loss_fn.forward(output.detach(), gen_attack.detach()).squeeze()

        g_loss.append(stats['opt'])
        g_mse.append(test_mse.item())
        g_fmse.append(feat_mse.item())
        g_psnr.append(test_psnr)
        g_lpips.append(test_lpips.item())

        plot(output)
        plt.title(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
                f"| PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} | LPIPS: {test_lpips:2.4f} |")
        plt.savefig(resultpath+'/attack_output'+str(i)+'.jpg')

g_loss = np.array(g_loss)
g_mse = np.array(g_mse)
g_fmse = np.array(g_fmse)
g_psnr = np.array(g_psnr)
g_lpips = np.array(g_lpips)
print(f"Rec.loss: {g_loss.mean():2.4f} +- {g_loss.std():2.4f}")
print(f"MSE: {g_mse.mean():2.4f} +- {g_mse.std():2.4f}")
print(f"F_MSE: {g_fmse.mean():2.4e} +- {g_fmse.std():2.4e}")
print(f"PSNR: {g_psnr.mean():4.2f} +- {g_psnr.std():4.2f}")
print(f"LPIPS: {g_lpips.mean():2.4f} +- {g_lpips.std():2.4f}")