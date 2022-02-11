import torch
from torchvision import transforms
import resnet
import os
from data_loader import get_cifar10_loader, GenDataset
from dataset_gen import generate_dataset, test
from torch.utils.data import DataLoader


batch_size = 128
modelpath = "./trained_models/CIFAR10_VAL_ResNet18_epoch_200.pt"
resultpath = "./valid_noisy_"+"001"+"/"

feature_loss = "l2"
logit_layer = False
weight = 0.01
noise_level = 0.001
start_point = "gaussian"
max_iterations = 300
inner_per_distance_record = 10
inner_per_image = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, valid_loader, test_loader = get_cifar10_loader(50, batch_size)

model = resnet.ResNet18()
checkpoint = torch.load(modelpath, map_location=device)
model.load_state_dict(checkpoint)
model = model.to(device)

if not os.path.exists(resultpath + "label.pt"):
        generate_dataset(model, test_loader, valid_loader, device, batch_size, resultpath,start_point = start_point, 
                loss_type=feature_loss, logit = logit_layer, weight = weight, noise_level = noise_level,
                max_iterations=max_iterations, inner_per_distance_record=inner_per_distance_record, 
                inner_per_image=inner_per_image)


testmodelpath = "./trained_models/last_CIFAR10_MASKED_ResNet18_noise0.001_results10.pt"
test_model = resnet.ResNet18()
test_model.load_state_dict(torch.load(testmodelpath, map_location=device))
test_model = test_model.to(device)

test(test_model, device, test_loader)

transform_valid = transforms.Compose([transforms.ToTensor(),])
noisy_test_data = GenDataset(resultpath, transform_valid)

noisy_test_loader = DataLoader(dataset=noisy_test_data,
                              batch_size=batch_size,
                              shuffle=True)

test(test_model, device, noisy_test_loader)



