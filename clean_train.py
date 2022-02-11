import trainmodel
import torch
from torchvision import transforms
import argparse

parser = argparse.ArgumentParser(description='Baseline')
parser.add_argument('--noise', type=float, default="0.0",
                    help='noise level used generate noisy image')
parser.add_argument('--starting', type=str, default="mixup",
                    help='The starting point. support type: gaussian, inclass, outclass')
parser.add_argument('--alpha', type=float, default="0.0",
                    help='The weight for the mixup, with 0.0 indicates original dataset') 
parser.add_argument('--result', type=str, default="",
                    help='The folder name suffix for results')
parser.add_argument('--defense', type=str, default="None",
                    help='The defense method: e.g., GradPrune_095')
parser.add_argument('--data', type=str, default="SVHN_TRAIN",
                    help='The dataset used for training. Default: SVHN_TRAIN')                    
            
args = parser.parse_args()

noise_level = args.noise
starting = args.starting
alpha = args.alpha
data = args.data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resultpath = "~/dataset"
transform_new = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])

# defense = 'GradPrune_095'
# defense = 'None'
defense = args.defense
print(defense)
resnet_type = 'ResNet18'
trainmodel.train(resnet_type, data, device, 200, data_path = resultpath, 
        transform = transform_new, write_accuracy=True, args = args, 
        save_model=False, defense=defense)
# trainmodel.train(resnet_type, 'CIFAR100', device, 200, data_path = resultpath, 
#         transform = transform_new, write_accuracy=True, args = args, 
#         save_model=False, defense=defense)
print('Parameters: '+defense+'\tModel: '+resnet_type)
