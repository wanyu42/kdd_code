import torch 
import trainmodel
import numpy as np
import argparse
from torchvision import transforms
from data_loader import get_cifar10_loader

if __name__=="__main__":
    np.random.seed(0)

    parser = argparse.ArgumentParser(description='Baseline for outclass 500 images')
    parser.add_argument('--noise', type=float, default="1e-2",
                        help='noise level used generate noisy image')
    parser.add_argument('--result', type=str, default="",
                        help='The folder name suffix for results') 
                
    args = parser.parse_args()

    noise_level = args.noise
    print("================================================================")
    print("================================================================")
    print("================================================================")

    transform_new = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])

    resultpath = "~/dataset"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainmodel.train('ResNet18', 'CIFAR10_Out', device, 100, 
        data_path = resultpath, transform = transform_new, noise = noise_level, write_accuracy = False, args = args, save_model = True)

    print("================================================================")
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    print("================================================================")



