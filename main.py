import torch

#import deeprobust.image.netmodels.train_model as trainmodel
import trainmodel
from torchvision import transforms
import argparse
from data_loader import *
from dataset_gen import *


# torch.manual_seed(0)
torch.manual_seed(123)

parser = argparse.ArgumentParser(description='Defense against Deep Leakage.')
parser.add_argument('--weight', type=float, default="1e-2",
                    help='the weight for image space difference.')
parser.add_argument('--noise', type=float,default="0.2",
                    help='the standard deviation of initial gaussian noise.')
parser.add_argument('--logit', type=str, default="False",
                    help='True for logit, False for second last layer output')
parser.add_argument('--loss', type=str, default="l2",
                    help='The feature distance metric. support type: l2, kl')  
parser.add_argument('--starting', type=str, default="mixup",
                    help='The starting point. support type: gaussian, inclass, outclass')
parser.add_argument('--alpha', type=float, default="1.0",
                    help='The weight for the mixup, with 0.0 indicates original dataset')
parser.add_argument('--random', type=str, default="False",
                    help='True with dropout, False without drop in generation process. default False')  
parser.add_argument('--train', type=str, default="False",
                    help='True with model.train(), False with model.eval(). default False')
parser.add_argument('--batch_stat', type=str, default="False",
                    help='True with batch stat matching, False without. default False')                       
parser.add_argument('--result', type=str, default="",
                    help='The folder name suffix for results')
parser.add_argument('--max_iter', type=int, default="1000",
                    help='The max iterations for data optimization')
parser.add_argument('--step_size', type=float, default="0.1",
                    help='The step size for dataset generation')
parser.add_argument('--pgd', type=str, default="True",
                    help='True: The optimization for dataset generation is pgd, False: standard gd')
args = parser.parse_args()


resnet_type = 'ResNet18'
# resnet_type = 'ResNet18CIFAR100'
# resnet_type = 'ResNet9'
# resnet_type = 'ResNet50'
swap_num = 128

originalDataPath="~/dataset"
# CIFAR10_VAL_ResNet18_epoch_200.pt for without dropout
# last_CIFAR10_VAL_ResNet18_noise0.01_alpha0.0_results.pt for with dropout
# for data generation process
if resnet_type == 'ResNet18':   
        modelpath = "./trained_models/CIFAR10_VAL_ResNet18_epoch_200.pt"
elif resnet_type == 'ResNet9':
        modelpath = './trained_models/last_CIFAR10_VAL_ResNet9_noise0.01_alpha0.0_results.pt'
elif resnet_type == 'ResNet50':
        modelpath = './trained_models/last_CIFAR10_VAL_ResNet50_noise0.0_alpha0.0_results.pt'
elif resnet_type == 'ResNet18CIFAR100':
        modelpath = './trained_models/best_CIFAR100_ResNet18None_noise0.0_alpha0.0_results.pt'
        
# modelpath = "./trained_models/last_CIFAR10_VAL_ResNet18_noise0.01_alpha0.0_results.pt"

weight = args.weight
noise_level = args.noise
start_point = args.starting
alpha = args.alpha
step_size = args.step_size

if start_point == "half":
        resultpath = "./results"+resnet_type+"swap"+str(swap_num)+args.result+"/"
else:
        resultpath = "./results"+resnet_type+args.result+"/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if(args.logit == "True"):
        logit_layer = True
else:
        logit_layer = False

if(args.random == "True"):
        dropout_random = True
else:
        dropout_random = False

if args.loss == "l2":
        feature_loss = "l2"
else:
        feature_loss = "kl"

if args.train == "True":
        model_train = True
else:
        model_train = False

if args.batch_stat == "True":
        batch_stat = True
else:
        batch_stat = False

if args.pgd == "True":
        pgd = True
else:
        pgd = False

print("================================================================")
print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
print("================================================================")

########################################################################
########################################################################
####################### Generate Dataset ###############################
########################################################################
########################################################################
#max_iterations = 400
max_iterations = args.max_iter
inner_per_image = 100
inner_per_distance_record = 10
batch_size = 128

transform_train = transforms.Compose([
        transforms.ToTensor(),
        ])
train_loader, _, test_loader = get_cifar10_loader(50, batch_size, transform_train)
transform_valid = transforms.Compose([transforms.ToTensor(),])
valid_dataset = CIFAR10(50, 0.0, mode = "valid", transform=transform_valid)
########################################################
##############  Load the trained model #################
########################################################
# dropout_random==True: image generation process will use dropout
if resnet_type == 'ResNet18':
        model = resnet.ResNet18(random=dropout_random)
elif resnet_type == 'ResNet9':
        model = resnet.ResNet9(random=dropout_random)
elif resnet_type == 'ResNet50':
        model = resnet.ResNet50(random=dropout_random)
elif resnet_type == 'ResNet18CIFAR100':
        model = resnet.ResNet18(num_classes=100, random=dropout_random)


checkpoint = torch.load(modelpath, map_location=device)
model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()
# test(model, device, test_loader)

# import ipdb; ipdb.set_trace()
# generate_dataset(model, train_loader,valid_loader, device, batch_size, resultpath, loss_type = feature_loss logit = logit_layer, weight = weight, noise_level = noise_level,
#         max_iterations=max_iterations, inner_per_distance_record=inner_per_distance_record, inner_per_image=inner_per_image)

if not os.path.exists(resultpath + "label.pt"):
        # generate_dataset(model, train_loader, valid_dataset, device, batch_size, resultpath, alpha = alpha, start_point = start_point, 
        #         loss_type=feature_loss, logit = logit_layer, weight = weight, noise_level = noise_level,
        #         max_iterations=max_iterations, inner_per_distance_record=inner_per_distance_record, 
        #         inner_per_image=inner_per_image, model_train=model_train, batch_stat=batch_stat)
        generate_dataset(model, train_loader, valid_dataset, device,batch_size,resultpath, alpha = alpha, start_point = start_point, 
                loss_type = feature_loss, weight = weight, noise_level = noise_level, swap_num=swap_num, conv=True,
                max_iterations=max_iterations, inner_per_distance_record=inner_per_distance_record, inner_per_image=inner_per_image)
else:
        print("Already exist dataset")

########################################################################
########################################################################
####################### Train new model ################################
########################################################################
########################################################################
print("================================================================")
print("================================================================")
print("================================================================")

transform_new = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

# defense = 'GradPrune_099'
defense = 'None'
# trainmodel.train('ResNet18', 'CIFAR10_MASKED', device, 100, data_path = resultpath, 
#         transform = transform_new, write_accuracy=True, args = args, 
#         save_model=False)
trainmodel.train(resnet_type, 'CIFAR10_MASKED', device, 200, data_path = resultpath, 
        transform = transform_new, write_accuracy=True, args = args, 
        save_model=False, defense=defense)


print("================================================================")
print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
print("================================================================")
print('Defense: '+defense)

