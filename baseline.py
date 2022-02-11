import torch
import argparse
import trainmodel
import numpy as np
from torchvision import transforms

if __name__=="__main__":
    np.random.seed(1)

    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('--noise', type=float, default="1e-2",
                        help='noise level used generate noisy image')
    parser.add_argument('--starting', type=str, default="gaussian",
                        help='The starting point. support type: gaussian, inclass, outclass')
    parser.add_argument('--alpha', type=float, default="0.0",
                        help='The weight for the mixup, with 0.0 indicates original dataset') 
    parser.add_argument('--result', type=str, default="",
                        help='The folder name suffix for results') 
                
    args = parser.parse_args()

    noise_level = args.noise
    starting = args.starting
    alpha = args.alpha
    # noise_level = 0.0

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

    if starting == "gaussian":
        # trainmodel.train('ResNet18', 'CIFAR10_Baseline', device, 100, 
        #     data_path = resultpath, transform = transform_new, noise = noise_level, 
        #     write_accuracy = True, args = args, save_model = False, regularize = False, alpha=alpha)
        trainmodel.train('ResNet9', 'CIFAR10_Baseline', device, 100, 
            data_path = resultpath, transform = transform_new, noise = noise_level, 
            write_accuracy = True, args = args, save_model = False, regularize = False, alpha=alpha)
    elif starting == "mixup":
        # trainmodel.train('ResNet18', 'CIFAR10_Mixup_base', device, 100, 
        #     data_path = resultpath, transform = transform_new, noise = noise_level, 
        #     write_accuracy = True, args = args, save_model = False, regularize = False, alpha=alpha)
        trainmodel.train('ResNet9', 'CIFAR10_Mixup_base', device, 100, 
            data_path = resultpath, transform = transform_new, noise = noise_level, 
            write_accuracy = True, args = args, save_model = False, regularize = False, alpha=alpha)

    print("================================================================")
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    print("================================================================")
