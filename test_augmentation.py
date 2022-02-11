from numpy import degrees
import torch
import argparse

from torchvision.transforms.transforms import RandomRotation
import trainmodel
from torchvision import transforms
import matplotlib.pyplot as plt
from data_loader import CIFAR10


torch.manual_seed(0)

parser = argparse.ArgumentParser(description='Train with different data augmentations.')
parser.add_argument('--weight', type=float, default="1e-2",
                    help='the weight for image space difference.')
parser.add_argument('--result', type=str, default="",
                    help='The folder name suffix for results')
parser.add_argument('--noise', type=float,default="0.2",
                    help='the standard deviation of initial gaussian noise.')           
args = parser.parse_args()

args.result = "data_aug_org_weight0_1outclassnoise0_2"
args.noise = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resultpath = "./resultsweight0_1outclassnoise0_2/"


# s=1
# color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
# transform_new = transforms.Compose([
#         transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomApply([color_jitter], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.GaussianBlur(kernel_size=int(0.1 * 32)),
#         transforms.ToTensor(),
#         ])

# transform_new = transforms.Compose([
#         # transforms.RandomCrop(32, padding=4),
#         transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
#         # transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(degrees = 30),
#         # transforms.GaussianBlur(kernel_size=int(0.1 * 32)),
#         transforms.ToTensor(),
#         ])
transform_new = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
# image_batch = CIFAR10(50, 0.0, mode = "train", transform=transform_new)
# tt = transforms.ToPILImage()
# plt.figure(figsize=(12, 8))
# for i in range(10):
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(tt(image_batch[i][0]))
#     plt.title("image=%d" % (i))
#     plt.axis('off')

# plt.show()

trainmodel.train('ResNet18', 'CIFAR10_MASKED', device, 100, data_path = resultpath, 
        transform = transform_new, write_accuracy=True, args = args, save_model=False)