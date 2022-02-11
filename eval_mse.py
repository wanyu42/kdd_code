import torch
from torchvision import transforms
from data_loader import CIFAR10, CompareDataset, MixupDatasetVerify, GaussianDatasetVerify
import csv


dropout = False
bn_modeified = False
noise_range = [0.004, 0.008, 0.02, 0.05, 0.1, 0.12, 0.13, 0.15, 0.17, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4]
# noise_range = [0.004, 0.008, 0.02, 0.05, 0.12, 0.13, 0.15, 0.17, 0.18, 0.25, 0.3, 0.35, 0.4]
# noise_range = [0.12, 0.13, 0.17, 0.18]
# noise_range = [0.2]
# baseline = "generated"
baseline = "baseline_noise"
# noise_level = 0.008
alpha_range = [0.0]
# alpha_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#alpha_range = [0.5]
# alpha_range = [0.75, 0.85]
weight = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trans = transforms.Compose([
            transforms.ToTensor(),
        ])
tt = transforms.ToPILImage()
mse_all = []

for noise_level in noise_range:
    for alpha_level in alpha_range:
        # noise_level = 0.1

        mse = 0
        sum_squared = 0.0
        count = 0

        if baseline == "generated":
            # datapath = './resultsweight0_01mixupnoise0_0alpha1_0/'
            # datapath = './resultsResNet9weight0_01mixupmax200noise0_2alpha1_0dropout'+str(dropout)+'train'+str(bn_modeified)+'batchFalse'+'/'
            # datapath = './resultsResNet9weight'+str(weight).replace('.','_')+'gaussianmax1000noise'+str(noise_level).replace('.', '_')+'alpha0_0dropout'+str(dropout)+'train'+str(bn_modeified)+'batchFalse'+'/'
            # datapath = './resultsweight0_0mixupnoise0_2/'
            # datapath = './resultsResNet9weight'+str(weight).replace('.','_')+'mixupmax100noise'+str(noise_level).replace('.', '_')+'alpha'+str(alpha_level).replace('.', '_')+'dropout'+str(dropout)+'train'+str(bn_modeified)+'batchFalse'+'/'
            datapath = './resultsResNet18weight'+str(weight).replace('.','_')+'mixupmax500noise'+str(noise_level).replace('.', '_')+'alpha'+str(alpha_level).replace('.', '_')+'dropout'+str(dropout)+'train'+str(bn_modeified)+'batchFalse'+'/'


            noisy_dataset = CompareDataset(datapath, trans)
            noisy_loader = torch.utils.data.DataLoader(noisy_dataset,
                            batch_size= 5000, shuffle=True)
            for idx, (noise, org, label) in enumerate(noisy_loader):
                noise, org, label = noise.to(device), org.to(device), label.to(device)
                mse = (noise-org).pow(2).mean().item()
                # import ipdb; ipdb.set_trace()
                break
        elif baseline == "baseline_noise":
            # for idx, (noise, org, label) in enumerate(noisy_loader):
            #     _, org, label = noise.to(device), org.to(device), label.to(device)
            #     noise = org + noise_level * torch.randn(org.size()).to(device)
            #     # import ipdb; ipdb.set_trace()
            #     noise_list = []
            #     for i in range(noise.shape[0]):
            #         noise_list.append(trans(tt(noise[i])))
            #     noise = torch.stack(noise_list)
            #     # import ipdb; ipdb.set_trace()
            #     mse = (noise-org).pow(2).mean().item()
            #     break
            noisy_dataset = GaussianDatasetVerify(50, noise_level, "train", trans)
            noisy_loader = torch.utils.data.DataLoader(noisy_dataset,
                        batch_size= 5000, shuffle=True)
            for idx, (noise, org, label) in enumerate(noisy_loader):
                noise, org, label = noise.to(device), org.to(device), label.to(device)
                mse = (noise-org).pow(2).mean().item()
                break

        elif baseline == "baseline_mixup":
            # mixup_data = MixupDataset(alpha, 50, transform_train=trans)
            # org = CIFAR10(50, 0.0, "train", transform=trans)
            # acc = 0.0
            # count = 0
            # for i in range(0, 10000, 10):
            #     acc += (mixup_data[i][0] - org[i][0]).pow(2).mean().item()
            #     count += 1
            # mse = acc/count
            mixup_data = MixupDatasetVerify(alpha_level, 50, transform_train=trans)
            mixup_loader = torch.utils.data.DataLoader(mixup_data,
                        batch_size= 5000, shuffle=True)
            acc = 0.0
            count = 0
            for idx, (noise, org, label) in enumerate(mixup_loader):
                noise, org, label = noise.to(device), org.to(device), label.to(device)
                acc += (noise-org).pow(2).mean().item()
                count += 1
                break
            mse = acc/count

        # print('mes: ', mse)
        mse_all.append([mse])

with open('mse.csv', 'w') as s:
    write = csv.writer(s)
    write.writerows(mse_all)
